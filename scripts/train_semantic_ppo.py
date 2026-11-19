"""Frozen-encoder semantic actor/critic PPO on one recorded on-policy batch.

Run exclusively on Hoffman2. This trains a focal slate selector, not an LLM
or a centralized multi-agent policy. F/test outcomes must not enter inputs.
"""

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

from scripts.train_hybrid import episode
from werewolf_agent.selector import FEATURES, Selector, design_row
from werewolf_agent.semantic_selector import (
    ARCHITECTURE, ENCODER_NAME, ENCODER_REVISION, FrozenEncoder,
    build_network, context_inputs, file_sha256, validate_encoder,
)


def ppo_surrogate(new_logp, old_logp, advantage, clip=.2):
    import torch
    ratio = torch.exp(new_logp - old_logp)
    return torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)


def episode_mean_sum(values, game_count):
    """All choices are already grouped into episodes; sum then divide by games."""
    if game_count < 1:
        raise ValueError("At least one game is required")
    return sum(values) / game_count


def distribution_kl(torch, records, model):
    values = []
    with torch.no_grad():
        for item in records:
            logits, _ = model(item["features"], item["labels_embedding"],
                              item["context_vector"], item["context_embedding"])
            current_logp = torch.log_softmax(logits, dim=0)
            values.append((item["old_probs"] * (item["old_full_logp"] - current_logp)).sum())
    stacked = torch.stack(values)
    return float(stacked.mean()), float(stacked.max())


def guarded_epoch(torch, model, actor_optimizer, critic_optimizer, records, game_count,
                  mean_limit=.02, max_limit=.1, entropy_coefficient=.001):
    """One full-batch step, with complete model/optimizer rollback on violation."""
    snapshot = (copy.deepcopy(model.state_dict()), copy.deepcopy(actor_optimizer.state_dict()),
                copy.deepcopy(critic_optimizer.state_dict()))

    def restore():
        model.load_state_dict(snapshot[0])
        actor_optimizer.load_state_dict(snapshot[1])
        critic_optimizer.load_state_dict(snapshot[2])
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)

    losses, entropies = [], []
    actor_optimizer.zero_grad(set_to_none=True)
    for item in records:
        logits, _ = model(item["features"], item["labels_embedding"],
                          item["context_vector"], item["context_embedding"])
        logp = torch.log_softmax(logits, dim=0)
        losses.append(-ppo_surrogate(logp[item["choice_index"]], item["old_selected_logp"], item["advantage"]))
        entropies.append(-(logp.exp() * logp).sum())
    policy_loss = episode_mean_sum(losses, game_count)
    entropy = episode_mean_sum(entropies, game_count)
    actor_loss = policy_loss - entropy_coefficient * entropy
    if not bool(torch.isfinite(actor_loss)):
        restore()
        return {"accepted": False, "reason": "nonfinite_actor_loss"}
    actor_loss.backward()
    actor_norm = torch.nn.utils.clip_grad_norm_(model.actor_parameters(), .5)
    if not bool(torch.isfinite(actor_norm)):
        restore()
        return {"accepted": False, "reason": "nonfinite_actor_gradient"}
    actor_optimizer.step()
    with torch.no_grad():
        # Preserve the historical bounded base-coefficient domain.
        model.base_weights.clamp_(-8., 8.)
    mean_kl, max_kl = distribution_kl(torch, records, model)
    if not math.isfinite(mean_kl) or not math.isfinite(max_kl) or mean_kl > mean_limit or max_kl > max_limit:
        restore()
        restored_mean, restored_max = distribution_kl(torch, records, model)
        return {"accepted": False, "reason": "kl_limit_or_nonfinite", "attempted_mean_kl": mean_kl,
                "attempted_max_kl": max_kl, "restored_mean_kl": restored_mean,
                "restored_max_kl": restored_max, "full_model_and_optimizers_restored": True}

    # The pooled actor representation is detached inside the model. Critic
    # optimization cannot change the policy or its accepted KL bound.
    critic_optimizer.zero_grad(set_to_none=True)
    critic_losses = []
    for item in records:
        _, value = model(item["features"], item["labels_embedding"],
                         item["context_vector"], item["context_embedding"])
        critic_losses.append((value.double() - item["return"]) ** 2)
    critic_loss = torch.stack(critic_losses).mean()
    if not bool(torch.isfinite(critic_loss)):
        restore()
        return {"accepted": False, "reason": "nonfinite_critic_loss",
                "full_model_and_optimizers_restored": True}
    critic_loss.backward()
    critic_norm = torch.nn.utils.clip_grad_norm_(model.critic.parameters(), .5)
    if not bool(torch.isfinite(critic_norm)):
        restore()
        return {"accepted": False, "reason": "nonfinite_critic_gradient",
                "full_model_and_optimizers_restored": True}
    critic_optimizer.step()
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
        restore()
        return {"accepted": False, "reason": "nonfinite_weights",
                "full_model_and_optimizers_restored": True}
    return {"accepted": True, "actor_loss": float(actor_loss.detach()),
            "policy_loss": float(policy_loss.detach()), "entropy_game_mean_sum": float(entropy.detach()),
            "critic_mse_decision_mean": float(critic_loss.detach()),
            "actor_gradient_norm_before_clip": float(actor_norm),
            "critic_gradient_norm_before_clip": float(critic_norm),
            "mean_kl": mean_kl, "max_kl": max_kl}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7391)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 32 or args.min_games < 2:
        parser.error("Use 1..32 epochs and at least two games")
    tensor_path = args.output.with_suffix(".pt")
    outputs = [args.output.resolve(), tensor_path.resolve(), args.report.resolve()]
    if len(set(outputs)) != 3 or args.start.resolve() in outputs or any(path.exists() for path in outputs):
        parser.error("Use three distinct new output paths; never replace a checkpoint")
    encoder_path = validate_encoder(args.encoder_path)
    old = Selector("hybrid_collect", args.start)
    games, exclusions, sources = {}, Counter(), {}
    for path in sorted(args.log_dir.rglob("*.jsonl")):
        item, reason = episode(path, old, args.generator_id)
        if reason:
            exclusions[reason] += 1
        else:
            terminal, _ = item
            match = terminal["match_hash"]
            if match in games:
                raise ValueError("Duplicate eligible focal seat in a training match")
            games[match] = item
            sources[match] = {"path": str(path.relative_to(args.log_dir)), "sha256": file_sha256(path)}
    if len(games) != args.min_games:
        raise ValueError(f"Expected exact predeclared batch of {args.min_games} games, found {len(games)}")

    import torch
    torch.manual_seed(args.seed)
    model = build_network(old.weights)
    encoder = FrozenEncoder(encoder_path)
    prepared, initial_errors, exact_base_checks = [], [], []
    role_counts, singleton_count = Counter(), 0
    for match, (terminal, choices) in games.items():
        role_counts[terminal["role_for_eval_only"]] += 1
        for choice in choices:
            features, labels = choice["features"], choice.get("candidate_labels")
            if not isinstance(labels, list) or len(labels) != len(features) or any(not isinstance(label, str) or not label for label in labels):
                raise ValueError("Sampled training choice lacks exact offered candidate labels")
            if len(features) < 2:
                singleton_count += 1
                continue
            context = {"day": choice["day"], "phase": choice["phase"]}
            values, text = context_inputs(features, context)
            embeddings = encoder.encode([*labels, text])
            with torch.no_grad():
                logits, old_value = model(features, embeddings[:-1], values, embeddings[-1])
                expected_logits = torch.tensor([sum(x * w for x, w in zip(design_row(row), old.weights))
                                                for row in features], dtype=torch.float64)
                exact_base_checks.append(bool(torch.equal(logits, expected_logits)))
                expected = torch.tensor(choice["probabilities"], dtype=torch.float64)
                actual = torch.softmax(logits, dim=0)
                if not torch.allclose(actual, expected, rtol=1e-7, atol=1e-9):
                    raise ValueError("Initial semantic policy is not the logged on-policy behavior")
                initial_errors.append(float((actual - expected).abs().max()))
                if float(old_value) != 0.:
                    raise ValueError("Initial critic must be exactly zero")
                old_logp = expected.log().detach().clone()
                target = torch.tensor(float(terminal["score"]), dtype=torch.float64)
                advantage = (target - old_value.double()).detach().clone()
            prepared.append({"match_hash": match, "features": features, "choice_index": choice["choice_index"],
                             "labels_embedding": embeddings[:-1].detach(), "context_vector": values,
                             "context_embedding": embeddings[-1].detach(), "old_probs": expected.detach().clone(),
                             "old_full_logp": old_logp, "old_selected_logp": old_logp[choice["choice_index"]].clone(),
                             "advantage": advantage, "return": target})
    if not prepared or not all(exact_base_checks):
        raise ValueError("Initial float64 logits must exactly reproduce the legacy design")
    if encoder._runtime is None or any(parameter.requires_grad for parameter in encoder._runtime[1].parameters()):
        raise ValueError("Semantic encoder is not frozen")

    actor_neural = [parameter for name, parameter in model.named_parameters()
                    if name != "base_weights" and not name.startswith("critic.")]
    actor_optimizer = torch.optim.Adam([{"params": [model.base_weights], "lr": .01},
                                        {"params": actor_neural, "lr": 3e-4}])
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    advantages_before = [item["advantage"].clone() for item in prepared]
    epochs = []
    for index in range(args.epochs):
        record = guarded_epoch(torch, model, actor_optimizer, critic_optimizer, prepared, len(games))
        epochs.append(dict(epoch=index + 1, **record))
        print(json.dumps(epochs[-1]), flush=True)
        if not record["accepted"]:
            break
    if any(not torch.equal(before, item["advantage"]) for before, item in zip(advantages_before, prepared)):
        raise ValueError("Fixed Monte Carlo advantages were modified")
    mean_kl, max_kl = distribution_kl(torch, prepared, model)
    if not math.isfinite(mean_kl) or not math.isfinite(max_kl) or mean_kl > .02 or max_kl > .1:
        raise ValueError("Final saved actor violates the training-state KL budget")
    train_settings = {
        "estimator": "focal_fixed_batch_categorical_ppo_mc_v1", "clip": .2,
        "base_lr": .01, "neural_actor_lr": 3e-4, "critic_lr": .001,
        "gradient_norm": .5, "entropy_coefficient": .001, "maximum_epochs": args.epochs,
        "actor_reduction": "sum_decision_surrogates_per_game_mean_over_games",
        "entropy_reduction": "sum_decision_entropies_per_game_mean_over_games",
        "critic_reduction": "decision_mean_mse", "advantage_normalization": False,
        "advantages": "fixed_terminal_score_minus_initial_critic_zero",
        "kl_direction": "old_logged_full_categorical_to_new", "kl_mean_limit": .02,
        "kl_max_limit": .1, "rollback": "entire_model_and_all_optimizer_states_then_halt",
        "base_coefficient_clip": [-8., 8.], "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tensor_path.open("xb") as stream:
        torch.save(model.state_dict(), stream)
    metadata = {
        "schema_version": 2, "features": FEATURES, "architecture": ARCHITECTURE,
        "encoder": {"name": ENCODER_NAME, "revision": ENCODER_REVISION,
                    "model_path": str(encoder_path.resolve()),
                    "files_sha256": {name: file_sha256(encoder_path / name)
                                     for name in ("config.json", "model.safetensors", "pytorch_model.bin",
                                                  "tokenizer.json", "vocab.txt", "tokenizer_config.json",
                                                  "special_tokens_map.json") if (encoder_path / name).is_file()}},
        "base_weights": model.base_weights.detach().tolist(),
        "weights_file": tensor_path.name, "weights_sha256": file_sha256(tensor_path),
        "parent": old.checkpoint_id, "parent_sha256": file_sha256(args.start),
        "generator_id": args.generator_id, "train_games": len(games),
        "train_match_hashes": sorted(games), "train_roles": dict(role_counts),
        "train_sampled_choices": len(prepared), "training": train_settings,
        "accepted_epochs": sum(item["accepted"] for item in epochs),
    }
    with args.output.open("x") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report = {
        "interpretation": "In-sample fixed-batch training diagnostics, not held-out playing strength.",
        "artifact": metadata, "epoch_history": epochs,
        "final_mean_kl": mean_kl, "final_max_kl": max_kl,
        "initial_logits_exact_legacy": all(exact_base_checks),
        "initial_probability_max_abs_error": max(initial_errors),
        "fixed_advantages_unchanged": True, "encoder_frozen": True,
        "excluded_seat_logs": dict(exclusions), "singleton_sampled_choices_skipped": singleton_count,
        "train_log_sources": sources, "log_dir": str(args.log_dir.resolve()),
        "optimizer_states_saved": False,
    }
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
