"""Report joint policy outcomes, proposal quality and delivery by focal game."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def read_rows(path):
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                return rows, True
            raise
    return rows, False


def sampled_generator(root):
    generator = None
    for path in sorted(root.rglob("*.jsonl")):
        rows, _ = read_rows(path)
        for row in rows:
            if row.get("event") != "hybrid_choice" or not row.get("sampled"):
                continue
            identity = row["generator_id"]
            if generator is not None and generator != identity:
                raise ValueError("Collection contains multiple sampled proposal distributions")
            generator = identity
    return generator


def report(root):
    games = []
    for path in sorted(root.rglob("*.jsonl")):
        rows, incomplete_tail = read_rows(path)
        choices = [row for row in rows if row.get("event") == "hybrid_choice"]
        assignment = next((row for row in rows if row.get("event") in {"comparison_assignment", "collection_assignment", "policy_assignment"}), {})
        if not choices and not assignment:
            continue
        head = choices[0] if choices else assignment
        sampled = [row for row in choices if row.get("sampled")]
        deliveries = Counter(row["decision_id"] for row in rows
                             if row.get("event") == "hybrid_delivery" and row.get("accepted"))
        terminals = [row for row in rows if row.get("event") == "game_result"]
        final = terminals[-1] if terminals else {}
        model_choices = [row for row in choices if row.get("sampled") or row.get("model_valid")]
        generator_head = model_choices[0] if model_choices else head
        games.append({
            "match_hash": head["match_hash"],
            "seat": head["seat"],
            "mode": head["mode"],
            "arm": assignment.get("arm"),
            "assignment_seed": assignment.get("assignment_seed"),
            "assignment_version": assignment.get("assignment_version"),
            "role": final.get("role_for_eval_only"),
            "score": final.get("score"),
            "termination_reason": final.get("termination_reason"),
            "terminal_count": len(terminals),
            "decisions": len(choices),
            "phase_counts": dict(Counter(row["phase"] for row in choices)),
            "sampled": len(sampled),
            "sampled_confirmed": sum(deliveries[row["decision_id"]] == 1 for row in sampled),
            "confirmed_decisions": sum(deliveries[row["decision_id"]] == 1 for row in choices),
            "delivery_errors": sum(bool(row.get("error_code")) for row in rows),
            "incomplete_tail": incomplete_tail,
            "model_fallbacks": sum(row["used_fallback"] for row in choices),
            "max_model_seconds": max((row["model_seconds"] for row in choices), default=None),
            "generator_id": generator_head.get("generator_id"),
            "generator_ids": sorted({row["generator_id"] for row in choices}),
            "checkpoint_id": head.get("checkpoint_id"),
            "opponent_pool": head.get("opponent_pool"),
        })
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--generator-id", action="store_true")
    modes.add_argument("--check-arms")
    parser.add_argument("--minimum-per-arm", type=int, default=12)
    args = parser.parse_args()
    if args.generator_id:
        print(sampled_generator(args.log_dir) or "no_sampled_proposals")
    else:
        games = report(args.log_dir)
        if args.check_arms:
            if args.minimum_per_arm < 1:
                parser.error("minimum-per-arm must be positive")
            arms = [arm.strip() for arm in args.check_arms.split(",")]
            counts = Counter(game["arm"] for game in games
                             if game["termination_reason"] == "completed" and game["terminal_count"] == 1)
            print(json.dumps(dict(counts)))
            sys.exit(0 if all(counts[arm] >= args.minimum_per_arm for arm in arms) else 1)
        print(json.dumps(games, indent=2))
