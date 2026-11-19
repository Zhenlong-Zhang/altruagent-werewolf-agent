"""Exercise two real HF processes and the actual competition selector path.

Synthetic latency/readiness evidence only. No tournament outcome is inferred.
Run within a Hoffman2 allocation with staged Qwen and MiniLM directories.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace


def run_worker(settings):
    semantic, checkpoint, log_dir, semantic_mode = settings
    os.environ["WEREWOLF_HYBRID_MODE"] = semantic_mode if semantic else "llm_select"
    os.environ["WEREWOLF_HYBRID_WEIGHTS"] = checkpoint
    os.environ["WEREWOLF_LOG_DIR"] = log_dir
    from werewolf_agent.competition_entry import create_agent
    from werewolf_agent.hybrid import Option
    from werewolf_agent.variants import LLMPolicy
    from altruagent import DecisionContext, LegalAction
    policy = create_agent() if semantic else LLMPolicy("direct")
    context = DecisionContext(session_id=f"synthetic-runtime-semantic-{semantic}", seat_position=6,
                              tournament_id=None, game_type="werewolf", agent_id="runtime-fixture")
    messages = [{"sender": i % 5, "content": (
        "I suspect Player1, but I have no verified check. Please compare the previous vote "
        "with the public claims. An accusation alone does not prove a role.")}
        for i in range(10)]
    view = dict(seat=6, role="villager", phase="day_vote", day=2,
                alive=list(range(7)), private_knowledge={6: "villager"},
                prior_ballots=[{"0": 1, "1": 2, "2": 1, "3": 2, "4": 1, "5": 2, "6": 0}],
                revealed_deaths=[], messages=messages)
    votes = [Option(str(i), f"Vote to eliminate Player{i}", i, 0.) for i in range(6)]
    wolf = dict(view, role="wolf", private_knowledge={6: "wolf", 5: "wolf"})
    rows = []
    try:
        for repetition in range(3):
            for name, current, options, speech in (
                ("vote_six", view, votes, False),
                ("wolf_speech", wolf, policy._speech(wolf, {}) if semantic else [], True),
            ):
                started = time.monotonic()
                if semantic:
                    selected = policy._select(context, current, options, Counter(), speech)
                    legal = selected in options
                else:
                    state = SimpleNamespace(
                        observation=f"Your role: {current['role'].upper()}", new_messages=[],
                        raw={"players": [{"position": i, "name": f"Player{i}"} for i in range(7)],
                             "game_state": {"day": 2, "phase": "day_vote",
                                            "alive": [f"Player{i}" for i in range(7)],
                                            "dead": [], "vote_history": []}},
                        legal_actions=[LegalAction(action_id=o.id, label=o.label, input={}) for o in votes],
                    )
                    answer, _, _ = policy._ask(state, context, "discussion" if speech else "action")
                    legal = (isinstance(answer.get("message"), str) if speech else
                             str(answer.get("action_id")) in {o.id for o in votes})
                elapsed = time.monotonic() - started
                record = json.loads(policy._logger.path.read_text().splitlines()[-1])
                row = {"scenario": name, "repetition": repetition, "total_seconds": elapsed,
                       "legal": legal,
                       **{key: record.get(key) for key in (
                           "checkpoint_id", "mode", "sampled", "model_valid", "model_error", "model_seconds",
                           "selector_error", "selector_seconds", "selector_deadline_kind", "used_fallback")}}
                if not semantic:
                    row.update(model_valid=not record["used_fallback"],
                               model_error=record.get("error_type"), model_seconds=record["elapsed_seconds"])
                rows.append(row)
    finally:
        if not semantic:
            policy.client.close()
        elif policy._model:
            policy._model.client.close()
    return {"semantic": semantic, "selector_mode": semantic_mode if semantic else "llm_direct", "rows": rows,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--semantic-mode", choices=("hybrid_collect", "hybrid_greedy"), default="hybrid_collect")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a new diagnostic output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = [(semantic, str(args.checkpoint.resolve()),
                 str(args.out.parent / f"runtime-fixture-{int(semantic)}"), args.semantic_mode)
                for semantic in (False, True)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        workers = list(pool.map(run_worker, settings))
    passed = all(row["legal"] and row["model_valid"] and not row["used_fallback"]
                 and row["total_seconds"] <= 30.0
                 for worker in workers for row in worker["rows"])
    passed = passed and all(row["mode"] == args.semantic_mode
                            and row["sampled"] == (args.semantic_mode == "hybrid_collect")
                            for worker in workers if worker["semantic"] for row in worker["rows"])
    result = {"interpretation": "Synthetic full-factory two-process readiness; not heldout win rate.",
              "passed": passed, "workers": workers,
              "cpu_threads_per_process": os.environ.get("WEREWOLF_CPU_THREADS")}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit("Semantic runtime readiness failed; do not create heldout matches")


if __name__ == "__main__":
    main()
