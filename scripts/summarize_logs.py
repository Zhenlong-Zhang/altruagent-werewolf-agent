#!/usr/bin/env python3
"""Summarize completed match outcomes from per-seat decision JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (math.nan, math.nan)
    p = wins / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return center - half, center + half


def read_matches(root: Path) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"seats": {}, "scores": {}}
    )
    for path in sorted(root.rglob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skip {path}: {exc}")
            continue
        for row in rows:
            match_id = row.get("match_hash")
            if not match_id:
                continue
            match = matches[match_id]
            seat = row.get("seat")
            if seat is not None and row.get("role_for_eval_only") in {"wolf", "seer", "villager"}:
                match["seats"][int(seat)] = row["role_for_eval_only"]
            if row.get("event") == "game_result" and seat is not None:
                score = row.get("score")
                if row.get("termination_reason") == "completed" and score in (-1, 1):
                    match["scores"][int(seat)] = score
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Directory containing decision JSONL files")
    args = parser.parse_args()
    matches = read_matches(args.log_dir)
    completed = [m for m in matches.values() if m["scores"].keys() & m["seats"].keys()]
    print(f"Observed match IDs: {len(matches)}")
    print(f"Matches with a terminal winner and role data: {len(completed)}")
    print(f"Incomplete matches: {len(matches) - len(completed)}")
    if not completed:
        print("No complete terminal outcomes found yet. Continue logging Testing matches.")
        return

    groups: dict[str, list[bool]] = defaultdict(list)
    for match in completed:
        for seat, score in match["scores"].items():
            role = match["seats"].get(seat)
            if role is not None:
                groups[role].append(score == 1)
                groups["overall"].append(score == 1)
    single_seat_games = all(len(m["scores"]) == 1 for m in completed)
    for group in ("overall", "wolf", "seer", "villager"):
        observations = groups.get(group, [])
        wins = sum(observations)
        if observations:
            interval = ""
            if single_seat_games:
                lo, hi = _wilson(wins, len(observations))
                interval = f" Wilson95=[{lo:.3f}, {hi:.3f}]"
            print(f"{group:10s} wins={wins}/{len(observations)} "
                  f"rate={wins/len(observations):.3f}{interval}")


if __name__ == "__main__":
    main()
