"""Audit a saved REINFORCE update and its in-sample policy-distribution effect.

Run on Hoffman2 after training. This is a read-only checkpoint audit: it
does not fit a model, alter weights, or estimate held-out match returns.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from scripts.train_hybrid import episode
from werewolf_agent.selector import FEATURES, WIDTH, Selector, design_row, probabilities


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    size = len(rows)
    return {
        "sampled_training_slates": size,
        "mean_kl_old_to_new": sum(row["kl"] for row in rows) / size if size else None,
        "max_kl_old_to_new": max((row["kl"] for row in rows), default=None),
        "mean_total_variation": sum(row["tv"] for row in rows) / size if size else None,
        "max_total_variation": max((row["tv"] for row in rows), default=None),
        "greedy_argmax_changed_count": sum(row["greedy_changed"] for row in rows),
        "greedy_argmax_changed_fraction": sum(row["greedy_changed"] for row in rows) / size if size else None,
    }


def analyze(log_dir, start_path, trained_path):
    record = json.loads(trained_path.read_text())
    old = Selector("hybrid_collect", start_path)
    new = Selector("hybrid_collect", trained_path)
    generator = record.get("generator_id")
    if not isinstance(generator, str) or not generator:
        raise ValueError("Trained artifact must name its collection generator_id")
    step = record.get("update_step")
    if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or not 0 < step <= 1:
        raise ValueError("Trained artifact has an invalid recorded update_step")
    recorded_matches = record.get("train_match_hashes")
    if (not isinstance(recorded_matches, list)
            or any(not isinstance(value, str) or not value for value in recorded_matches)
            or len(set(recorded_matches)) != len(recorded_matches)):
        raise ValueError("Trained artifact must list distinct nonempty training match hashes")

    games, sources, excluded = {}, {}, Counter()
    for path in sorted(log_dir.rglob("*.jsonl")):
        item, reason = episode(path, old, generator)
        if reason:
            excluded[reason] += 1
            continue
        terminal, choices = item
        match = terminal["match_hash"]
        if match in games:
            raise ValueError(f"Multiple eligible sampled focal logs for match {match}")
        games[match] = (terminal, choices)
        sources[match] = {"path": str(path.relative_to(log_dir)), "sha256": digest(path)}
    if not games:
        raise ValueError("No eligible sampled training games found")

    # Independent implementation of the recorded update: one score-function
    # gradient sum, terminal +/-1 reward, average by games, then coordinate clip.
    gradient = [0.0] * WIDTH
    roles = Counter()
    for terminal, choices in games.values():
        roles[terminal["role_for_eval_only"]] += 1
        for choice in choices:
            rows = [design_row(features) for features in choice["features"]]
            selected = rows[choice["choice_index"]]
            for index in range(WIDTH):
                expected = sum(mass * row[index] for mass, row in zip(choice["probabilities"], rows))
                gradient[index] += terminal["score"] * (selected[index] - expected)
    reproduced = [max(-8., min(8., weight + step * value / len(games)))
                  for weight, value in zip(old.weights, gradient)]
    residuals = [abs(actual - expected) for actual, expected in zip(new.weights, reproduced)]
    eligible_matches = set(games)
    recorded_set = set(recorded_matches)
    checks = {
        "feature_schema_matches": tuple(record["features"]) == FEATURES,
        "parent_checkpoint_matches": record.get("parent") == old.checkpoint_id,
        "estimator_matches": record.get("estimator") == "on_policy_reinforce",
        "eligible_match_hash_set_matches": eligible_matches == recorded_set,
        "eligible_match_count_matches": record.get("train_games") == len(games) == len(recorded_matches),
        "recorded_role_counts_match": record.get("train_roles") == dict(roles),
        "recomputed_weights_match": all(value <= 1e-12 for value in residuals),
    }

    slate_metrics, by_role, by_kind = [], defaultdict(list), defaultdict(list)
    for terminal, choices in games.values():
        for choice in choices:
            before = probabilities(choice["features"], old.weights)
            after = probabilities(choice["features"], new.weights)
            old_greedy = max(range(len(before)), key=before.__getitem__)
            new_greedy = max(range(len(after)), key=after.__getitem__)
            metric = {
                "kl": sum(p * (math.log(p) - math.log(q)) for p, q in zip(before, after)),
                "tv": .5 * sum(abs(p - q) for p, q in zip(before, after)),
                "greedy_changed": old_greedy != new_greedy,
            }
            slate_metrics.append(metric)
            by_role[terminal["role_for_eval_only"]].append(metric)
            by_kind["speech" if choice.get("phase") == "speech" else "action"].append(metric)
    delta = [after - before for before, after in zip(old.weights, new.weights)]
    names = [feature if condition == "intercept" else f"{feature} * {condition}"
             for condition in ("intercept", *FEATURES[4:]) for feature in FEATURES[:4]]
    return {
        "analysis": "saved_update_reproduction_and_in_sample_policy_distribution_change",
        "interpretation": "Training-slate diagnostics, not held-out return or evidence of improved win rate.",
        "metric_weighting": "Equal weight per sampled eligible training slate; argmax ties use first offered index.",
        "audit_passed": all(checks.values()), "checks": checks,
        "inputs": {"log_dir": str(log_dir), "start": str(start_path) if start_path else "built_in_initial_weights",
                   "start_sha256": digest(start_path) if start_path else None,
                   "trained": str(trained_path), "trained_sha256": digest(trained_path)},
        "generator_id": generator, "parent_checkpoint_id": old.checkpoint_id,
        "trained_checkpoint_id": new.checkpoint_id, "recorded_update_step": step,
        "eligible_games": len(games), "eligible_roles": dict(roles),
        "eligible_match_hashes": sorted(games), "recorded_match_hashes": sorted(recorded_matches),
        "missing_recorded_matches": sorted(recorded_set - eligible_matches),
        "extra_eligible_matches": sorted(eligible_matches - recorded_set),
        "excluded_seat_logs": dict(excluded), "eligible_log_sources": sources,
        "coefficient_change": {
            "changed_count_at_1e_minus_12": sum(abs(value) > 1e-12 for value in delta),
            "total_coefficients": WIDTH, "l2_delta": math.sqrt(sum(value * value for value in delta)),
            "max_absolute_delta": max(map(abs, delta)),
            "max_reproduction_error": max(residuals),
            "coefficients": [dict(name=name, before=before, after=after, delta=change,
                                  reproduced=recomputed)
                             for name, before, after, change, recomputed
                             in zip(names, old.weights, new.weights, delta, reproduced)],
        },
        "in_sample_training_slate_metrics": summarize(slate_metrics),
        "in_sample_by_role": {role: summarize(rows) for role, rows in sorted(by_role.items())},
        "in_sample_by_kind": {kind: summarize(rows) for kind, rows in sorted(by_kind.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        parser.error("log_dir must be an existing directory")
    protected = {args.trained.resolve()}
    if args.start:
        protected.add(args.start.resolve())
    if args.out.resolve() in protected:
        parser.error("Audit output must not replace an input checkpoint")
    result = analyze(log_dir, args.start, args.trained)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit_passed": result["audit_passed"],
                      "checks": result["checks"],
                      "coefficient_change": {key: value for key, value in result["coefficient_change"].items()
                                             if key != "coefficients"},
                      "in_sample_training_slate_metrics": result["in_sample_training_slate_metrics"]}))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
