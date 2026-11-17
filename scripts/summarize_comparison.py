"""Join a frozen practice manifest to official focal-seat results."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(manifest, directory):
    output = []
    for match_id, (seat, variant) in manifest.items():
        match_hash = hashlib.sha256(match_id.encode()).hexdigest()[:16]
        path = directory / f"match_{match_hash}_seat_{seat}.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        finished = [event for event in events if event["event"] == "game_result"]
        result = finished[-1] if finished else {}
        calls = [event for event in events if event["event"] == "model_call"]
        actions = [event for event in calls if event["task"] == "action"]
        speech = [event for event in calls if event["task"] == "discussion"]
        output.append({
            "match_id": match_id,
            "seat": seat,
            "candidate": variant,
            "strategy": result.get("strategy_version"),
            "role": result.get("role_for_eval_only"),
            "score": result.get("score"),
            "termination": result.get("termination_reason"),
            "model_actions": sum(not event["used_fallback"] for event in actions),
            "action_fallbacks": sum(event["used_fallback"] for event in actions),
            "speech_fallbacks": sum(event["used_fallback"] for event in speech),
            "max_model_seconds": max((event["elapsed_seconds"] for event in calls), default=None),
            "rejected_actions": sum(bool(event.get("error_code")) for event in events),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    rows = summarize(json.loads(args.manifest.read_text()), args.log_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
