"""Audit conditional old-vs-recorded-NN greedy choices on observed G slates.

Read-only, standard library plus the legacy Selector. No Torch, encoder,
training, alternative trajectories or counterfactual game outcomes.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from werewolf_agent.selector import Selector


PARENT_ID = "d7ebf7670444067a"
NN_ID = "6ca4df8fd98821e6"
GUARD = "verified-innocent-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    count = len(rows)
    changes = sum(row["changed"] for row in rows)
    return {"eligible_decisions": count, "changed_decisions": changes,
            "changed_fraction": changes / count if count else None,
            "games_with_eligible_decisions": len({row["match_hash"] for row in rows}),
            "games_with_any_change": len({row["match_hash"] for row in rows if row["changed"]})}


def analyze(log_dir, summary_path, artifact_path, old_path):
    summary = json.loads(summary_path.read_text())
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)
    old = Selector("hybrid_greedy", old_path)
    tensor = artifact_path.parent / artifact["weights_file"]
    tensor_hash = sha(tensor)
    nn_id = hashlib.sha256(raw + tensor_hash.encode()).hexdigest()[:16]
    problems, missing, sources = [], [], []

    def problem(error, match=None):
        problems.append({"error": error, **({"match_hash": match} if match else {})})

    if (artifact.get("schema_version") != 2 or artifact.get("parent") != old.checkpoint_id
            or old.checkpoint_id != PARENT_ID or artifact.get("parent_sha256") != sha(old_path)):
        problem("old_checkpoint_does_not_match_recorded_NN_parent")
    if tensor_hash != artifact.get("weights_sha256") or nn_id != NN_ID:
        problem("NN_checkpoint_fingerprint_mismatch")
    if summary.get("assignment_version") != "wolf-policy-v1":
        problem("unexpected_G_assignment_version")
    if summary.get("seed") != "r03-semantic-greedy-heldout-v1":
        problem("unexpected_G_assignment_seed")
    if summary.get("protocol_problems"):
        problem("summary_reports_protocol_problems")
    games = {}
    all_matches = set()
    for game in summary.get("games", []):
        match = game.get("match_hash")
        if not isinstance(match, str) or not match or match in all_matches:
            problem("missing_or_duplicate_summary_match")
            continue
        all_matches.add(match)
        if game.get("assigned_arm") == "nn_greedy":
            games[match] = game
    overlap = sorted(all_matches & set(artifact.get("train_match_hashes", [])))
    if overlap:
        problem("E_G_training_evaluation_overlap")
    if not games:
        missing.append("no_assigned_nn_greedy_games")

    logs = defaultdict(list)
    for path in sorted(log_dir.rglob("*.jsonl")):
        lines = path.read_text().splitlines()
        try:
            rows = [json.loads(line) for line in lines if line.strip()]
        except json.JSONDecodeError:
            problem("malformed_or_truncated_actual_game_log")
            continue
        if any(not isinstance(row, dict) for row in rows):
            problem("non_object_log_row")
            continue
        hashes = {row.get("match_hash") for row in rows}
        seats = {row.get("seat") for row in rows}
        if len(hashes) != 1 or len(seats) != 1:
            problem("mixed_or_missing_log_identity")
            continue
        match, seat = next(iter(hashes)), next(iter(seats))
        if match not in all_matches:
            problem("log_outside_controller_summary", match)
            continue
        if match in games and seat == games[match].get("assigned_seat"):
            logs[match].append(rows)
            sources.append({"path": str(path.relative_to(log_dir)), "match_hash": match, "sha256": sha(path)})

    counts, metrics, by_role, by_phase = Counter(), [], defaultdict(list), defaultdict(list)
    per_game = []
    for match, game in games.items():
        if len(logs.get(match, [])) != 1:
            missing.append({"match_hash": match, "reason": "missing_or_duplicate_NN_focal_log"})
            if len(logs.get(match, [])) > 1:
                problem("duplicate_NN_focal_log", match)
            continue
        rows = logs[match][0]
        assignments = [row for row in rows if row.get("event") == "policy_assignment"]
        if len(assignments) != 1 or any(assignments[0].get(key) != expected for key, expected in
                                       (("arm", "nn_greedy"), ("mode", "hybrid_greedy"),
                                        ("checkpoint_id", NN_ID), ("evidence_guard", GUARD),
                                        ("seat", game["assigned_seat"]), ("assignment_seed", summary.get("seed")))):
            problem("NN_focal_assignment_mismatch", match)
        choices = [row for row in rows if row.get("event") == "hybrid_choice"]
        if len(choices) != game.get("decisions", 0):
            problem("summary_choice_count_mismatch", match)
        ids = [row.get("decision_id") for row in choices]
        if None in ids or len(set(ids)) != len(ids):
            problem("duplicate_or_missing_decision_id", match)
        terminals = [row for row in rows if row.get("event") == "game_result"]
        role = game.get("role")
        if (game.get("terminal_count") != 1 or game.get("termination_reason") != "completed"
                or len(terminals) != 1):
            missing.append({"match_hash": match, "reason": "NN_game_not_resolved"})
        elif terminals[0].get("role_for_eval_only") != role:
            problem("summary_terminal_role_mismatch", match)
        role = role or "unknown"
        game_metrics = []
        for choice in choices:
            counts["all_NN_focal_choices"] += 1
            if choice.get("mode") != "hybrid_greedy" or choice.get("checkpoint_id") != NN_ID:
                counts["excluded_invalid_mode_or_checkpoint"] += 1
                problem("choice_not_recorded_NN_greedy_policy", match)
                continue
            features, candidates = choice.get("features"), choice.get("candidate_ids")
            if (not isinstance(features, list) or not isinstance(candidates, list)
                    or len(features) != len(candidates) or not features or len(set(candidates)) != len(candidates)):
                counts["excluded_invalid_slate"] += 1
                problem("invalid_recorded_candidate_slate", match)
                continue
            forced = len(features) < 2
            fallback = bool(choice.get("used_fallback"))
            counts["forced_singleton_choices"] += forced
            counts["used_fallback_choices"] += fallback
            counts["selector_error_choices"] += bool(choice.get("selector_error"))
            counts["model_invalid_choices_including_forced"] += not bool(choice.get("model_valid"))
            if forced:
                counts["excluded_forced_singleton"] += 1
                continue
            if fallback or choice.get("selector_error"):
                counts["excluded_fallback"] += 1
                continue
            if not choice.get("model_valid"):
                counts["excluded_no_valid_model_proposal"] += 1
                continue
            config = choice.get("proposer_configuration", {})
            if (config.get("proposer") != "ids" or config.get("evidence_guard_version") != GUARD
                    or config.get("actual_dtype") != "bfloat16" or config.get("actual_cpu_threads") != 4):
                counts["excluded_wrong_guard_or_cold_configuration"] += 1
                problem("eligible_NN_slate_not_guarded_warm_proposal", match)
                continue
            selected = choice.get("choice_index")
            if (type(selected) is not int or not 0 <= selected < len(features)
                    or choice.get("sampled") is not False
                    or choice.get("probabilities") != [float(index == selected) for index in range(len(features))]):
                counts["excluded_invalid_actual_greedy_selection"] += 1
                problem("recorded_NN_selection_not_actual_greedy_onehot", match)
                continue
            try:
                old_index, _ = old.choose(features)
            except (ValueError, TypeError):
                counts["excluded_invalid_legacy_features"] += 1
                problem("legacy_comparator_cannot_read_exact_logged_features", match)
                continue
            metric = {"match_hash": match, "decision_id": choice["decision_id"], "role": role,
                      "phase": choice.get("phase", "unknown"), "changed": old_index != selected,
                      "old_guarded_argmax_index": old_index, "recorded_NN_argmax_index": selected,
                      "old_guarded_candidate_id": candidates[old_index],
                      "recorded_NN_candidate_id": candidates[selected]}
            metrics.append(metric)
            game_metrics.append(metric)
            by_role[role].append(metric)
            by_phase[metric["phase"]].append(metric)
        per_game.append({"match_hash": match, "role": role, **aggregate(game_metrics)})
    overall = aggregate(metrics)
    if not metrics:
        interpretation = "No eligible observed NN states; conditional action identity cannot be assessed."
    elif overall["changed_decisions"] == 0:
        interpretation = ("Conditional actions were identical on all eligible observed NN states. "
                          "Do not attribute a higher observed win rate to different NN-head choices on these states. "
                          "Forced/fallback/excluded decisions remain separately counted; this is not a whole-game counterfactual.")
    else:
        interpretation = ("Recorded NN and old guarded greedy choices differed on some identical observed slates. "
                          "This establishes conditional decision changes, not that either alternative would win more games.")
    return {
        "audit_integrity_passed": not problems, "observed_NN_logs_complete": not missing,
        "interpretation": interpretation, "protocol_issues": problems, "incomplete_reasons": missing,
        "assigned_NN_games": len(games), "NN_games_with_eligible_decisions": overall["games_with_eligible_decisions"],
        "NN_games_with_any_changed_decision": overall["games_with_any_change"],
        "changed_game_hashes": sorted({row["match_hash"] for row in metrics if row["changed"]}),
        "unchanged_eligible_game_hashes": sorted({row["match_hash"] for row in metrics}
                                                - {row["match_hash"] for row in metrics if row["changed"]}),
        "counts": dict(counts), "conditional_comparison": overall,
        "by_role": {key: aggregate(value) for key, value in sorted(by_role.items())},
        "by_phase": {key: aggregate(value) for key, value in sorted(by_phase.items())},
        "per_NN_game": per_game, "eligible_decision_comparisons": metrics,
        "comparator": {"old_checkpoint_id": old.checkpoint_id, "NN_checkpoint_id": nn_id,
                       "state": "Exact recorded before-decision guarded candidate features/order; first-index greedy ties."},
        "provenance": {"helper_sha256": sha(Path(__file__)), "summary_sha256": sha(summary_path),
                       "old_sha256": sha(old_path), "NN_metadata_sha256": sha(artifact_path),
                       "NN_tensor_sha256": tensor_hash, "E_G_overlap_hashes": overlap,
                       "NN_focal_log_sha256_manifest": sources},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.log_dir.is_dir() or args.out.exists() or args.out.resolve() in {
            args.summary.resolve(), args.training_artifact.resolve(), args.old.resolve()}:
        parser.error("Use an actual G decisions directory and a new exclusive output path")
    try:
        report = analyze(args.log_dir, args.summary, args.training_artifact, args.old)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        report = {"audit_integrity_passed": False, "observed_NN_logs_complete": False,
                  "protocol_issues": [{"error": "unreadable_or_invalid_comparison_input",
                                       "exception_type": type(exc).__name__}]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("audit_integrity_passed", "observed_NN_logs_complete")}))
    raise SystemExit(1 if not report["audit_integrity_passed"] else 0 if report["observed_NN_logs_complete"] else 2)


if __name__ == "__main__":
    main()
