"""Report joint policy outcomes, proposal quality and delivery by focal game."""

import argparse
from collections import Counter
import json
from pathlib import Path


def report(root):
    games = []
    for path in sorted(root.rglob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        choices = [row for row in rows if row.get("event") == "hybrid_choice"]
        if not choices:
            continue
        sampled = [row for row in choices if row.get("sampled")]
        deliveries = Counter(row["decision_id"] for row in rows
                             if row.get("event") == "hybrid_delivery" and row.get("accepted"))
        final = next((row for row in reversed(rows) if row.get("event") == "game_result"), {})
        games.append({
            "match_hash": choices[0]["match_hash"],
            "seat": choices[0]["seat"],
            "mode": choices[0]["mode"],
            "role": final.get("role_for_eval_only"),
            "score": final.get("score"),
            "termination_reason": final.get("termination_reason"),
            "decisions": len(choices),
            "phase_counts": dict(Counter(row["phase"] for row in choices)),
            "sampled": len(sampled),
            "sampled_confirmed": sum(deliveries[row["decision_id"]] == 1 for row in sampled),
            "model_fallbacks": sum(row["used_fallback"] for row in choices),
            "max_model_seconds": max(row["model_seconds"] for row in choices),
            "generator_id": choices[0]["generator_id"],
            "checkpoint_id": choices[0]["checkpoint_id"],
            "opponent_pool": choices[0].get("opponent_pool"),
        })
    return games


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.log_dir), indent=2))
