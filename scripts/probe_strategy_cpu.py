"""Operational CPU fallback probe; separate from the frozen paired GPU screen."""

import argparse
import json
from pathlib import Path
import time

from scripts.bench_strategy_models import scenarios
from werewolf_agent.strategy_backend import HFGenerationBackend
from werewolf_agent.strategy_candidates import (
    build_candidate_request, parse_strategy_answer, validate_action_answer,
    validate_speech_candidates,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    backend = HFGenerationBackend(args.model_path, args.model_id, args.revision,
                                   device="cpu", dtype="bfloat16", threads=16)
    backend.load()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # One private-fact action and one typed speech completion, not a win test.
    selected = [row for row in scenarios() if row["shift"] == 0
                and not row["order_reversed"]
                and row["scenario"] in {"verified_wolf_over_claim", "defend_verified_nonwolf"}]
    with args.out.open("x") as stream:
        for index, row in enumerate(selected):
            result = {"scope": "operational CPU latency/format only; not the GPU screen",
                      "scenario": row["scenario"], "model_id": args.model_id,
                      "revision": args.revision, "seed": 98001 + index,
                      "seconds_cap": 90., "max_new_tokens": 2048,
                      "valid": False, "error": None}
            started = time.monotonic()
            try:
                completion = backend.complete(build_candidate_request(row["packet"], row["task"]),
                                              2048, .6, 98001 + index, started + 90.)
                result["completion"] = completion
                answer = parse_strategy_answer(completion["text"],
                                                finish_reason=completion["provenance"]["finish_reason"])
                if row["task"] == "action":
                    result["parsed"] = validate_action_answer(answer, row["packet"])
                else:
                    result["parsed"] = validate_speech_candidates(answer, row["packet"])
                result["valid"] = True
            except Exception as exc:
                result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["elapsed_seconds"] = time.monotonic() - started
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    main()
