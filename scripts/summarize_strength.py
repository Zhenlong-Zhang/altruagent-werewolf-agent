"""Summarize assigned comparison games, including absent and unresolved workers."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from scripts.report_hybrid import read_rows, report


ROLE_WEIGHTS = {"villager": 4 / 7, "seer": 1 / 7, "wolf": 2 / 7}
NO_RESULT_REASONS = {"testing_stale", "tournament_stale", "cancelled", "testing_cancelled", "tournament_cancelled"}


def wilson(wins, total):
    if not total:
        return None
    z = 1.959963984540054
    rate = wins / total
    divisor = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / divisor
    width = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / divisor
    return [max(0, center - width), min(1, center + width)]


def expected_assignment(session_id, arms, seed, version="wolf-compare-v1"):
    """Independently check the runtime's frozen routing rule against UI receipts."""
    digest = hashlib.sha256(f"{version}:{seed}:{session_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 7, arms[int.from_bytes(digest[8:16], "big") % len(arms)]


def group_summary(games):
    resolved = [game for game in games if game.get("terminal_count") == 1
                and game.get("score") in (-1, 0, 1)
                and game.get("termination_reason") == "completed"]
    wins = sum(game["score"] > 0 for game in resolved)
    roles = {}
    for role in ROLE_WEIGHTS:
        selected = [game for game in resolved if game.get("role") == role]
        won = sum(game["score"] > 0 for game in selected)
        roles[role] = dict(games=len(selected), wins=won,
                           win_rate=won / len(selected) if selected else None,
                           wilson95=wilson(won, len(selected)))
    standardized = None
    if all(roles[role]["games"] for role in ROLE_WEIGHTS) and sum(r["games"] for r in roles.values()) == len(resolved):
        standardized = sum(ROLE_WEIGHTS[role] * roles[role]["win_rate"] for role in ROLE_WEIGHTS)
    unresolved = len(games) - len(resolved)
    return dict(assigned=len(games), resolved=len(resolved), unresolved=unresolved, wins=wins,
                win_rate_resolved=wins / len(resolved) if resolved else None,
                wilson95=wilson(wins, len(resolved)), roles=roles,
                seat_counts=dict(Counter(game.get("assigned_seat") for game in games)),
                role_standardized_win_rate=standardized,
                all_assigned_win_bounds=[wins / len(games), (wins + unresolved) / len(games)] if games else None,
                selected_decisions=sum(game.get("decisions", 0) for game in games),
                confirmed_decisions=sum(game.get("confirmed_decisions", 0) for game in games),
                model_fallbacks=sum(game.get("model_fallbacks", 0) for game in games),
                delivery_errors=sum(game.get("delivery_errors", 0) for game in games),
                termination_reasons=dict(Counter(game.get("termination_reason") or "unresolved" for game in games)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--seed", default="r03-frozen-v1")
    parser.add_argument("--arms", default="ids,reasoned")
    parser.add_argument("--assignment-version", default="wolf-compare-v1",
                        choices=("wolf-compare-v1", "wolf-collect-v1", "wolf-selector-v1", "wolf-policy-v1"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",")]
    receipts = json.loads(args.census.read_text())
    observed = {}
    for game in report(args.log_dir):
        if game["match_hash"] in observed:
            raise ValueError("Multiple focal logs for one comparison match")
        observed[game["match_hash"]] = game
    games = []
    problems = []
    enrolled = set()
    for receipt in receipts:
        session = receipt.get("game_session_id") or receipt["match_id"]
        fingerprint = hashlib.sha256(session.encode()).hexdigest()[:16]
        if fingerprint in enrolled:
            raise ValueError("Duplicate controller match receipt")
        enrolled.add(fingerprint)
        seat, arm = expected_assignment(session, arms, args.seed, args.assignment_version)
        game = observed.pop(fingerprint, dict(match_hash=fingerprint))
        if "arm" in game and (game["arm"] != arm or game["seat"] != seat):
            problems.append(dict(match_hash=fingerprint, error="assignment_mismatch"))
        if game.get("assignment_version") is not None and game["assignment_version"] != args.assignment_version:
            problems.append(dict(match_hash=fingerprint, error="assignment_version_mismatch"))
        if game.get("assignment_seed") is not None and game["assignment_seed"] != args.seed:
            problems.append(dict(match_hash=fingerprint, error="assignment_seed_mismatch"))
        game.update(assigned_arm=arm, assigned_seat=seat)
        game["controller_receipt_late"] = bool(receipt.get("receipt_recorded_after_terminal_observation"))
        games.append(game)
    for game in observed.values():
        problems.append(dict(match_hash=game["match_hash"], error="missing_controller_receipt_or_session_mapping"))
        game["assigned_arm"] = game["arm"]
        games.append(game)
    configurations = {}
    configurations_by_arm = {arm: {} for arm in arms}
    arm_by_hash = {game["match_hash"]: game["assigned_arm"] for game in games}
    for path in args.log_dir.rglob("*.jsonl"):
        rows, _ = read_rows(path)
        for row in rows:
            if row.get("event") == "hybrid_choice" and row.get("model_valid"):
                config = row.get("proposer_configuration", {})
                configurations[json.dumps(config, sort_keys=True)] = config
                arm = arm_by_hash.get(row.get("match_hash"))
                if arm in configurations_by_arm:
                    configurations_by_arm[arm][json.dumps(config, sort_keys=True)] = config
    output = dict(seed=args.seed, receipt_count=len(receipts), games=games, protocol_problems=problems,
                  assignment_version=args.assignment_version,
                  late_receipt_count=sum(game.get("controller_receipt_late", False) for game in games),
                  arms={arm: group_summary([game for game in games if game["assigned_arm"] == arm]) for arm in arms},
                  observed_model_configurations=list(configurations.values()),
                  model_configurations_by_arm={arm: list(values.values()) for arm, values in configurations_by_arm.items()},
                  interpretation="Prospective roster-specific results. Report unresolved assignments and role imbalance before promotion.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["arms"]))


if __name__ == "__main__":
    main()
