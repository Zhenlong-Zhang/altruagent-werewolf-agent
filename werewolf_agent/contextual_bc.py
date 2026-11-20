"""Game-disjoint teacher imitation and training-only value warm start."""

import copy
import hashlib
import math

from .contextual_selector import tensorize
from .vendor.event_ppo import masked_distribution


def warm_start(network, encoder, episodes, *, epochs, learning_rate, dev_fraction, seed, chunk_size):
    import torch
    if (type(epochs) is not int or epochs < 1 or type(chunk_size) is not int or chunk_size < 1
            or type(seed) is not int or not 0 < dev_fraction < .5
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Explicit finite BC training/split settings required")
    usable = []
    identifiers = set()
    for episode in episodes:
        if episode["game_id"] in identifiers:
            raise ValueError("Duplicate demonstration game")
        identifiers.add(episode["game_id"])
        if episode["checkpoint_id"] is not None or not episode["complete"] or episode["terminal_return"] not in {-1., 1.}:
            raise ValueError("Use verified complete teacher games for this initial warm start")
        rows = []
        for demonstration in episode["demonstrations"]:
            snapshot, index = demonstration["serialized"], demonstration["teacher_index"]
            if (demonstration["game_id"] != episode["game_id"] or snapshot["owner"] != episode["owner"]
                    or type(index) is not int or not 0 <= index < len(snapshot["candidate_mask"])
                    or not snapshot["candidate_mask"][index]):
                raise ValueError("Unbound/invalid actual teacher target")
            rows.append(dict(demonstration, terminal_return=episode["terminal_return"]))
        if rows:
            usable.append((episode["game_id"], rows))
    if len(usable) < 2:
        raise ValueError("Need at least two demonstration games for a disjoint development split")
    ordered = sorted(usable, key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}".encode()).hexdigest())
    dev_count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * dev_fraction)))
    development, training = ordered[:dev_count], ordered[dev_count:]
    train_rows = [row for _, records in training for row in records]
    dev_rows = [row for _, records in development for row in records]
    train_counts = {identifier: len(records) for identifier, records in training}
    dev_counts = {identifier: len(records) for identifier, records in development}
    actor_optimizer = torch.optim.Adam(network.actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(network.critic.parameters(), lr=learning_rate)

    def metrics(rows, counts, learn=False):
        total_ce, total_value, correct, count = 0., 0., 0, 0
        for offset in range(0, len(rows), chunk_size):
            chunk = rows[offset:offset + chunk_size]
            inputs = tensorize([row["serialized"] for row in chunk], encoder)
            logits, values = network(inputs)
            distribution = masked_distribution(logits, inputs["candidate_mask"])
            targets = torch.tensor([row["teacher_index"] for row in chunk])
            weights = torch.tensor([1. / (len(counts) * counts[row["game_id"]]) for row in chunk])
            ce = (-distribution.log_prob(targets) * weights).sum()
            loss = (.5 * (values - torch.tensor([row["terminal_return"] for row in chunk])).square() * weights).sum()
            if not bool(torch.isfinite(ce + loss)):
                raise FloatingPointError("Nonfinite BC objective")
            if learn:
                (ce + loss).backward()
            total_ce += float(ce.detach()); total_value += float(loss.detach())
            correct += int((distribution.probs.argmax(dim=-1) == targets).sum()); count += len(chunk)
        return {"game_mean_cross_entropy": total_ce, "game_mean_value_loss": total_value,
                "teacher_choice_accuracy": correct / max(count, 1), "decisions": count}

    network.eval()
    history, best, best_ce, best_epoch = [], None, math.inf, None
    with torch.no_grad():
        baseline = metrics(dev_rows, dev_counts)
    for epoch in range(epochs):
        actor_optimizer.zero_grad(set_to_none=True); critic_optimizer.zero_grad(set_to_none=True)
        train_metrics = metrics(train_rows, train_counts, learn=True)
        for parameters in (list(network.actor.parameters()), list(network.critic.parameters())):
            if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
                raise FloatingPointError("Nonfinite BC gradient")
            torch.nn.utils.clip_grad_norm_(parameters, .5)
        actor_optimizer.step(); critic_optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in network.parameters()):
            raise FloatingPointError("Nonfinite BC parameter")
        with torch.no_grad():
            dev_metrics = metrics(dev_rows, dev_counts)
        history.append({"epoch": epoch + 1, "training": train_metrics, "development": dev_metrics})
        if dev_metrics["game_mean_cross_entropy"] < best_ce:
            best_ce, best_epoch = dev_metrics["game_mean_cross_entropy"], epoch + 1
            best = copy.deepcopy({"network": network.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
                                  "critic_optimizer": critic_optimizer.state_dict()})
    network.load_state_dict(best["network"])
    # Optimizer moments correspond to the selected epoch, not the last epoch.
    report = {"algorithm": "actual-teacher BC plus training-only faction value warm start",
              "selection": "minimum game-mean development teacher cross entropy; earliest tie",
              "split_seed": seed, "dev_fraction": dev_fraction, "epochs": epochs,
              "learning_rate": learning_rate, "chunk_size": chunk_size, "selected_epoch": best_epoch,
              "training_game_hashes": [hashlib.sha256(identifier.encode()).hexdigest()[:16] for identifier, _ in training],
              "development_game_hashes": [hashlib.sha256(identifier.encode()).hexdigest()[:16] for identifier, _ in development],
              "initial_development": baseline, "history": history,
              "scope": "Pre-PPO C0 imitation, not a win-rate evaluation or LLM weight update"}
    return report, {key: best[key] for key in ("actor_optimizer", "critic_optimizer")}
