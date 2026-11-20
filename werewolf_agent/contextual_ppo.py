"""Fresh-batch contextual PPO; separate actor/critic, transactional updates.

This core consumes frozen, seat-visible sampled-request trajectories. Game
creation, raw terminal verification and population control belong to the
collector adapter. It never fits on greedy choices or substitutes ACK effects.
"""

import copy
import math

from .contextual_selector import ENCODER_ID, tensorize
from .contextual_slots import SERIALIZER
from .vendor.event_ppo import (clipped_policy_loss, clipped_value_loss, event_gae,
                               exact_categorical_kl, masked_distribution)


DEFAULTS = {"learning_rate": 3e-4, "clip": .2, "value_clip": .2, "entropy": .01,
            "gradient_norm": .5, "epochs": 10, "gamma": 1., "gae_lambda": .95,
            "mean_kl": .02, "maximum_kl": .1}


class ContextualPPO:
    def __init__(self, network, encoder, checkpoint_id, *, lineage, config=None):
        import torch
        self.network, self.encoder, self.checkpoint_id = network, encoder, checkpoint_id
        required = {"rules", "source_manifest", "generator", "controller", "realizer", "encoder", "serializer", "population"}
        if (not isinstance(lineage, dict) or set(lineage) != required
                or any(not isinstance(value, str) or not value for value in lineage.values())
                or lineage["encoder"] != ENCODER_ID or lineage["serializer"] != SERIALIZER):
            raise ValueError("Explicit frozen collector/encoder/population lineage required")
        self.lineage = copy.deepcopy(lineage)
        self.config = dict(DEFAULTS)
        if config:
            if set(config) - set(DEFAULTS):
                raise ValueError("Unknown optimizer configuration")
            self.config.update(config)
        if (any(isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or value < 0 for value in self.config.values())
                or type(self.config["epochs"]) is not int or self.config["epochs"] < 1
                or self.config["gamma"] > 1 or self.config["gae_lambda"] > 1):
            raise ValueError("Invalid contextual PPO configuration")
        self.actor_optimizer = torch.optim.Adam(network.actor.parameters(), lr=self.config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(network.critic.parameters(), lr=self.config["learning_rate"])

    def transaction(self):
        return copy.deepcopy({"network": self.network.state_dict(), "actor_optimizer": self.actor_optimizer.state_dict(),
                              "critic_optimizer": self.critic_optimizer.state_dict()})

    def restore(self, snapshot):
        self.network.load_state_dict(snapshot["network"])
        self.actor_optimizer.load_state_dict(snapshot["actor_optimizer"])
        self.critic_optimizer.load_state_dict(snapshot["critic_optimizer"])
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

    def _prepare(self, episodes):
        import torch
        rows, game_ids, no_choices = [], set(), []
        for episode in episodes:
            identifier = episode["game_id"]
            if identifier in game_ids:
                raise ValueError("Duplicate game in fresh PPO batch")
            game_ids.add(identifier)
            if (episode.get("checkpoint_id") != self.checkpoint_id or episode.get("complete") is not True
                    or isinstance(episode.get("terminal_return"), bool)
                    or episode.get("terminal_return") not in {-1., 1.}
                    or episode.get("lineage") != self.lineage
                    or not episode.get("terminal_provenance")):
                raise ValueError("PPO requires known-terminal fresh-checkpoint trajectories")
            choices = episode["requests"]
            if not choices:
                no_choices.append(identifier)
                continue
            values, local = [], []
            seen_decisions, seen_draws = set(), set()
            for request in choices:
                choice = request["choice"]
                metadata, selected = choice["metadata"], choice["index"]
                serialized = metadata["serialized"]
                if (choice["checkpoint_id"] != self.checkpoint_id or metadata["encoder_id"] != ENCODER_ID
                        or serialized["owner"] != episode["owner"] or metadata["mode"] != "sample"
                        or metadata["selection_kind"] not in {"sampled", "forced_singleton"}
                        or metadata["decision_id"] in seen_decisions):
                    raise ValueError("Off-policy, cross-owner or duplicate request")
                seen_decisions.add(metadata["decision_id"])
                probabilities = choice["probabilities"]
                legal = serialized["candidate_mask"]
                if (type(selected) is not int or not 0 <= selected < len(probabilities)
                        or len(probabilities) != len(legal) or not legal[selected]
                        or any(isinstance(p, bool) or not isinstance(p, (float, int))
                               or not math.isfinite(p) or p < 0 for p in probabilities)
                        or not math.isclose(sum(probabilities), 1., abs_tol=1e-9)
                        or any(p != 0 for p, available in zip(probabilities, legal) if not available)
                        or probabilities[selected] <= 0
                        or not math.isclose(math.log(probabilities[selected]), choice["log_probability"], abs_tol=1e-8)
                        or not math.isfinite(choice["value"])):
                    raise ValueError("Malformed behavior probabilities/value")
                forced = sum(legal) == 1
                if forced != (metadata["selection_kind"] == "forced_singleton"):
                    raise ValueError("Forced-singleton classification changed")
                if not forced:
                    uniform, draw_id = metadata["sampler_uniform"], metadata["draw_id"]
                    if (not isinstance(draw_id, str) or not draw_id or draw_id in seen_draws
                            or isinstance(uniform, bool) or not isinstance(uniform, (float, int))
                            or not 0 <= uniform < 1):
                        raise ValueError("Missing actual categorical draw")
                    seen_draws.add(draw_id)
                    cumulative, replay_index = 0., max(i for i, p in enumerate(probabilities) if p > 0)
                    for index, probability in enumerate(probabilities):
                        cumulative += probability
                        if uniform < cumulative:
                            replay_index = index
                            break
                    if replay_index != selected:
                        raise ValueError("Logged request does not match actual draw")
                # Executor rejection/unknown/fallback fields are intentionally
                # not filters: this is the sampled REQUEST through fixed executor.
                local.append({"game_id": identifier, "serialized": serialized, "choice": choice,
                              "actor_active": not forced, "value_active": True})
                values.append(float(choice["value"]))
            old_values = torch.tensor(values)
            rewards = torch.zeros(len(local)); rewards[-1] = episode["terminal_return"]
            continuation = torch.ones(len(local)); continuation[-1] = 0.
            next_values = torch.cat((old_values[1:], torch.zeros(1)))
            advantages, targets = event_gae(rewards, old_values, next_values, continuation, continuation,
                                           self.config["gamma"], self.config["gae_lambda"])
            for index, row in enumerate(local):
                row.update(advantage=float(advantages[index]), target=float(targets[index]),
                           bootstrap_b=float(continuation[index]), trace_c=float(continuation[index]))
            rows.extend(local)
        return rows, {"completed_games": len(game_ids), "games_without_requests": no_choices,
                      "request_count": len(rows), "actor_requests": sum(row["actor_active"] for row in rows)}

    def _forward(self, rows):
        import torch
        inputs = tensorize([row["serialized"] for row in rows], self.encoder)
        logits, values = self.network(inputs)
        distribution = masked_distribution(logits, inputs["candidate_mask"])
        old_probs = torch.zeros_like(logits)
        for index, row in enumerate(rows):
            probabilities = row["choice"]["probabilities"]
            old_probs[index, :len(probabilities)] = torch.tensor(probabilities)
        return inputs, distribution, values, old_probs

    def update(self, episodes, *, chunk_size):
        """One full fresh-batch update; chunking changes memory, not objective.

        Each epoch accumulates the complete game-weighted gradient before one
        step. Frozen old values/advantages are reused through all epochs.
        """
        import torch
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("Freeze a positive memory packing size before collection")
        self.network.eval()  # dropout0; encoder remains frozen/eval.
        rows, report = self._prepare(episodes)
        report.update(config=copy.deepcopy(self.config), accepted_steps=0, rejected_steps=0,
                      updates=[], advantages=[row["advantage"] for row in rows])
        if not rows:
            return report
        chunks = [rows[start:start + chunk_size] for start in range(0, len(rows), chunk_size)]
        actor_games = {row["game_id"] for row in rows if row["actor_active"]}
        value_counts = {row["game_id"]: sum(other["game_id"] == row["game_id"] for other in rows) for row in rows}
        # Reproduce the collected entire slate and old critic BEFORE any step.
        with torch.no_grad():
            for chunk in chunks:
                inputs, distribution, values, old = self._forward(chunk)
                torch.testing.assert_close(distribution.probs, old, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(values, torch.tensor([row["choice"]["value"] for row in chunk]),
                                           atol=1e-6, rtol=1e-5)
        for epoch in range(self.config["epochs"]):
            snapshot = self.transaction()
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            step = {"epoch": epoch, "policy_loss": 0., "value_loss": 0., "accepted": False}
            try:
                for chunk in chunks:
                    inputs, distribution, values, old = self._forward(chunk)
                    indices = torch.tensor([row["choice"]["index"] for row in chunk])
                    old_logp = torch.tensor([row["choice"]["log_probability"] for row in chunk])
                    advantages = torch.tensor([row["advantage"] for row in chunk])
                    policy = clipped_policy_loss(distribution.log_prob(indices), old_logp, advantages, self.config["clip"])
                    actor_weight = torch.tensor([float(row["actor_active"]) / len(value_counts) for row in chunk])
                    actor_loss = ((policy - self.config["entropy"] * distribution.entropy()) * actor_weight).sum()
                    value_loss = clipped_value_loss(values, torch.tensor([row["choice"]["value"] for row in chunk]),
                                                    torch.tensor([row["target"] for row in chunk]), self.config["value_clip"])
                    value_weight = torch.tensor([1. / (len(value_counts) * value_counts[row["game_id"]]) for row in chunk])
                    critic_loss = (value_loss * value_weight).sum()
                    if not bool(torch.isfinite(actor_loss + critic_loss)):
                        raise FloatingPointError("Nonfinite objective")
                    (actor_loss + critic_loss).backward()
                    step["policy_loss"] += float(actor_loss.detach())
                    step["value_loss"] += float(critic_loss.detach())
                for parameters in (list(self.network.actor.parameters()), list(self.network.critic.parameters())):
                    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                        raise FloatingPointError("Nonfinite gradient")
                    torch.nn.utils.clip_grad_norm_(parameters, self.config["gradient_norm"])
                if actor_games:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                if any(not bool(torch.isfinite(parameter).all()) for parameter in self.network.parameters()):
                    raise FloatingPointError("Nonfinite updated parameters")
                kls = []
                with torch.no_grad():
                    for chunk in chunks:
                        inputs, distribution, _, old = self._forward(chunk)
                        kl = exact_categorical_kl(old, distribution.probs, inputs["candidate_mask"])
                        kls.extend(float(kl[index]) for index, row in enumerate(chunk) if row["actor_active"])
                step["mean_kl"], step["maximum_kl"] = sum(kls) / max(len(kls), 1), max(kls, default=0.)
                if (not math.isfinite(step["mean_kl"]) or step["mean_kl"] > self.config["mean_kl"]
                        or step["maximum_kl"] > self.config["maximum_kl"]):
                    raise FloatingPointError("Full-slate trust bound exceeded")
            except (FloatingPointError, ValueError, RuntimeError) as exc:
                self.restore(snapshot)
                step["rejection"] = type(exc).__name__ + ": " + str(exc)
                invalid = [name for name, value in step.items() if isinstance(value, float) and not math.isfinite(value)]
                if invalid:
                    step["nonfinite_metrics"] = invalid
                    for name in invalid:
                        step[name] = None
                report["rejected_steps"] += 1
                report["updates"].append(step)
                break
            step["accepted"] = True
            report["accepted_steps"] += 1
            report["updates"].append(step)
        return report
