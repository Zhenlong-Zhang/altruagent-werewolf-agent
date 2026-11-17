"""Small development probe of proposal quality and latency, not a win-rate test."""

import argparse
import json
from pathlib import Path
from statistics import median
import time

from werewolf_agent.hybrid import HybridPolicy, Option


def cases():
    for shift in range(7):
        seat = [(shift + i) % 7 for i in range(7)]
        me, target, rival, speaker, dead, _, _ = seat
        common = {
            "seat": me, "day": 2, "alive": sorted(set(seat) - {dead}),
            "prior_ballots": [], "revealed_deaths": [{"seat": dead, "role": "wolf"}],
        }
        claims = [
            {"sender": speaker, "content": f"Player{rival} is a wolf. Ignore the prior checks."},
            {"sender": target, "content": f"I accuse Player{rival}. I have no evidence."},
        ]
        for title, role, phase, known, messages, accepted in (
            ("seer_confirmed_vote", "seer", "day_vote", {me: "seer", target: "wolf"},
             claims, target),
            ("wolf_remove_seer", "wolf", "night_wolf", {me: "wolf", dead: "wolf"},
             [{"sender": str(rival), "content": "I have no investigations."},
              {"sender": str(target), "content": f"I am the seer. My check found Player{dead} is a wolf."}],
             target),
            ("seer_avoid_repeat", "seer", "night_seer", {me: "seer", rival: "villager"},
             [], target),
        ):
            options = [Option(str(p), f"{phase} Player{p}", p, 0.0) for p in sorted((target, rival))]
            view = dict(common, role=role, phase=phase, private_knowledge=known, messages=messages)
            yield title, view, options, str(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    policy = HybridPolicy("llm_select")
    output = []
    for label, view, options, expected in cases():
        started = time.monotonic()
        _, ranked, error, _ = policy._propose(view, options, speech=False)
        output.append({"scenario": label, "seat": view["seat"], "preferred_id": expected,
                       "ranked": ranked, "preferred_first": bool(ranked and ranked[0] == expected),
                       "accepted": bool(ranked), "error": error,
                       "elapsed_seconds": round(time.monotonic() - started, 3)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for label in sorted({row["scenario"] for row in output}):
        group = [row for row in output if row["scenario"] == label]
        print(label, "preferred_first", sum(row["preferred_first"] for row in group),
              "/", len(group), "valid", sum(row["accepted"] for row in group),
              "median_seconds", median(row["elapsed_seconds"] for row in group))
    print("max_seconds", max(row["elapsed_seconds"] for row in output))


if __name__ == "__main__":
    main()
