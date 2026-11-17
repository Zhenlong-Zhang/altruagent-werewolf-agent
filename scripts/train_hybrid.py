"""One on-policy REINFORCE update on a frozen batch of official focal seat logs."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from werewolf_agent.selector import FEATURES, Selector, WIDTH, design_row, probabilities


def episode(path, selector, generator_id):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    choices = [row for row in rows if row.get("event") == "hybrid_choice"
               and row.get("mode") == "hybrid_collect" and row.get("sampled") is True]
    if not choices:
        return None, "no_sampled_choices"
    final = [row for row in rows if row.get("event") == "game_result"]
    if len(final) != 1:
        return None, "missing_or_duplicate_terminal"
    terminal = final[0]
    if terminal.get("termination_reason") != "completed" or terminal.get("score") not in (-1, 1):
        return None, "invalid_terminal"
    if any(row.get("error_code") for row in rows):
        return None, "delivery_error"
    ids = [choice.get("decision_id") for choice in choices]
    if None in ids or len(ids) != len(set(ids)):
        return None, "duplicate_or_missing_decision_id"
    delivery = [
        row for row in rows
        if row.get("event") == "hybrid_delivery" and row.get("decision_id") in ids
    ]
    if Counter(row["decision_id"] for row in delivery if row.get("accepted") is True) != Counter(ids):
        return None, "unconfirmed_sampled_delivery"
    for row in choices:
        if (row.get("checkpoint_id") != selector.checkpoint_id
                or row.get("generator_id") != generator_id
                or row.get("match_hash") != terminal.get("match_hash")
                or row.get("seat") != terminal.get("seat")):
            return None, "checkpoint_or_episode_mismatch"
        features = row["features"]
        if len(features) < 2 or not 0 <= row["choice_index"] < len(features):
            return None, "invalid_choice"
        expected = probabilities(features, selector.weights)
        actual = row["probabilities"]
        if len(actual) != len(expected) or any(
            not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)
            for a, b in zip(expected, actual)
        ):
            return None, "propensity_mismatch"
    return (terminal, choices), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--start", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-id", required=True)
    parser.add_argument("--min-games", type=int, default=30)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    if args.min_games < 2 or not 0 < args.step <= 1:
        parser.error("Use at least two games and an update step between zero and one")
    selector = Selector("hybrid_collect", args.start)
    excluded = Counter()
    games = {}
    for path in sorted(args.log_dir.rglob("*.jsonl")):
        item, reason = episode(path, selector, args.generator_id)
        if reason:
            excluded[reason] += 1
        else:
            terminal, _ = item
            match = terminal["match_hash"]
            if match in games:
                raise SystemExit(f"More than one sampled focal seat in match {match}")
            games[match] = item
    print(json.dumps({"usable_games": len(games), "excluded_seat_logs": excluded}))
    if len(games) < args.min_games:
        raise SystemExit("Insufficient completed on-policy games; weights were not written")
    gradient = [0.0] * WIDTH
    roles = Counter()
    for terminal, choices in games.values():
        roles[terminal["role_for_eval_only"]] += 1
        for choice in choices:
            slate = [design_row(row) for row in choice["features"]]
            selected = slate[choice["choice_index"]]
            for j, value in enumerate(selected):
                mean = sum(prob * row[j] for prob, row in zip(choice["probabilities"], slate))
                gradient[j] += terminal["score"] * (value - mean)
    weights = [
        max(-8., min(8., w + args.step * g / len(games)))
        for w, g in zip(selector.weights, gradient)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema_version": 1, "features": FEATURES, "weights": weights,
                   "parent": selector.checkpoint_id, "train_games": len(games),
                   "generator_id": args.generator_id,
                   "train_roles": roles, "estimator": "on_policy_reinforce",
                   "train_match_hashes": sorted(games)}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
