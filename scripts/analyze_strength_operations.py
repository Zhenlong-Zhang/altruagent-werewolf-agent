"""Read-only final F cohort integrity/operations audit (standard library only).

Exit 0: integrity passed and all assigned games resolved with >=12 per arm.
Exit 1: protocol/configuration/hash integrity failed. Exit 2: integrity passed
but the cohort is incomplete. Operational failures are reported, never used
to discard assigned games or choose a winning policy.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


PARENT_ID = "d7ebf7670444067a"
ADAPTED_ID = "6ca4df8fd98821e6"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GUARD = "verified-innocent-v1"
PROFILES = {
    "incumbent": ("hybrid_greedy", "none", PARENT_ID),
    "guarded": ("hybrid_greedy", GUARD, PARENT_ID),
    "sampled": ("hybrid_collect", GUARD, PARENT_ID),
    "adapted": ("hybrid_collect", GUARD, ADAPTED_ID),
}
POOL = ["direct", "rules", "rules", "rules", "rules", "rules"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def latency(values):
    return {"count": len(values), "median_seconds": quantile(values, .5),
            "p95_seconds": quantile(values, .95), "max_seconds": max(values, default=None),
            "quantile_definition": "linear interpolation at (n-1)*q"}


def read_rows(path):
    content = path.read_text()
    lines = content.splitlines()
    rows, incomplete = [], False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not content.endswith("\n"):
                incomplete = True
                break
            raise
        if not isinstance(row, dict):
            raise ValueError("Non-object event row")
        rows.append(row)
    return rows, incomplete


def generator(config, guarded, warm=False):
    payload = {key: config[key] for key in
               ("proposer", "prompt_version", "requested_dtype", "device", "requested_revision")}
    if "requested_cpu_threads" in config:
        payload["requested_cpu_threads"] = str(config["requested_cpu_threads"])
    if guarded:
        payload["evidence_guard_version"] = GUARD
    payload["actual_dtype"] = config["requested_dtype"] if warm else config["actual_dtype"]
    payload["actual_revision"] = config["actual_revision"]
    if "requested_cpu_threads" in config:
        payload["actual_cpu_threads"] = config["requested_cpu_threads"] if warm else config["actual_cpu_threads"]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    prefix = "hybrid-slate-v1" + ("-verified-innocent-v1" if guarded else "")
    return prefix + "-cfg-" + digest + ":Qwen3-8B"


def analyze(log_dir, summary_path, artifact_path):
    summary = json.loads(summary_path.read_text())
    artifact_raw = artifact_path.read_bytes()
    artifact = json.loads(artifact_raw)
    tensor = artifact_path.parent / artifact["weights_file"]
    tensor_hash = sha(tensor)
    adapted_id = hashlib.sha256(artifact_raw + tensor_hash.encode()).hexdigest()[:16]
    issues, missing, manifest = [], [], []

    def issue(code, match=None, **details):
        issues.append(dict(error=code, **({"match_hash": match} if match else {}), **details))

    if artifact.get("schema_version") != 2 or artifact.get("parent") != PARENT_ID:
        issue("training_artifact_schema_or_parent_mismatch")
    if tensor_hash != artifact.get("weights_sha256") or adapted_id != ADAPTED_ID:
        issue("adapted_checkpoint_fingerprint_mismatch")
    if summary.get("assignment_version") != "wolf-policy-v1":
        issue("summary_assignment_version_mismatch")
    if summary.get("protocol_problems"):
        issue("summary_reports_protocol_problems", count=len(summary["protocol_problems"]))
    if summary.get("late_receipt_count", 0):
        issue("controller_receipts_recorded_late", count=summary["late_receipt_count"])
    games = {}
    for game in summary.get("games", []):
        match, arm = game.get("match_hash"), game.get("assigned_arm")
        if not isinstance(match, str) or not match or arm not in PROFILES:
            issue("malformed_summary_game")
            continue
        if match in games:
            issue("duplicate_summary_match", match)
        games[match] = game
    if not games:
        missing.append("no_assigned_games")
    if summary.get("receipt_count") != len(games):
        issue("controller_receipt_count_mismatch")
    overlap = sorted(set(games) & set(artifact.get("train_match_hashes", [])))
    if overlap:
        issue("E_training_F_evaluation_overlap", count=len(overlap))

    by_match = defaultdict(list)
    for path in sorted(log_dir.rglob("*.jsonl")):
        manifest.append({"path": str(path.relative_to(log_dir)), "sha256": sha(path)})
        try:
            rows, incomplete = read_rows(path)
        except (ValueError, json.JSONDecodeError):
            issue("malformed_log", log_file=str(path.relative_to(log_dir)))
            continue
        hashes = {row.get("match_hash") for row in rows}
        if len(hashes) != 1 or None in hashes:
            issue("mixed_or_missing_log_match_identity", log_file=str(path.relative_to(log_dir)))
            continue
        match = next(iter(hashes))
        if match not in games:
            issue("log_not_in_controller_summary", match)
            continue
        if incomplete:
            issue("incomplete_log_tail", match)
        by_match[match].append((path, rows))

    arm_counts, resolved_counts = Counter(), Counter()
    counters = {arm: Counter() for arm in PROFILES}
    timings = {arm: defaultdict(list) for arm in PROFILES}
    direct_errors = {arm: Counter() for arm in PROFILES}
    selector_errors = {arm: Counter() for arm in PROFILES}
    proposer_errors = {arm: Counter() for arm in PROFILES}
    generation_counts = {arm: Counter() for arm in PROFILES}
    config_counts = {arm: Counter() for arm in PROFILES}
    direct_models = {arm: Counter() for arm in PROFILES}

    def add_time(arm, key, value, match):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            issue("invalid_logged_latency", match, field=key)
        else:
            timings[arm][key].append(value)

    for match, game in games.items():
        arm = game["assigned_arm"]
        arm_counts[arm] += 1
        mode, guard, checkpoint = PROFILES[arm]
        seat = game.get("assigned_seat")
        if type(seat) is not int or seat not in range(7):
            issue("invalid_assigned_seat", match)
            continue
        if game.get("controller_receipt_late"):
            issue("game_controller_receipt_late", match)
        resolved = (game.get("terminal_count") == 1 and game.get("score") in (-1, 0, 1)
                    and game.get("termination_reason") == "completed")
        if resolved:
            resolved_counts[arm] += 1
        else:
            missing.append(dict(match_hash=match, reason="unresolved_assigned_game"))
        focal_rows, assignment_rows, focal_files = [], [], 0
        seat_files, direct_call_count = Counter(), 0
        direct_seat = next(player for player in range(7) if player != seat)
        for _, rows in by_match.get(match, []):
            seats = {row.get("seat") for row in rows}
            if len(seats) != 1:
                issue("mixed_log_seat_identity", match)
                continue
            row_seat = next(iter(seats))
            seat_files[row_seat] += 1
            if row_seat == seat:
                focal_files += 1
                focal_rows.extend(rows)
                assignment_rows.extend(row for row in rows if row.get("event") == "policy_assignment")
            for row in rows:
                if row.get("event") != "model_call":
                    continue
                if row_seat != direct_seat:
                    issue("model_call_outside_expected_direct_opponent_seat", match)
                    continue
                direct_call_count += 1
                count = counters[arm]
                count["direct_model_calls"] += 1
                count["direct_response_received"] += bool(row.get("response_received"))
                count["direct_used_fallback"] += bool(row.get("used_fallback"))
                count["direct_valid_output"] += not bool(row.get("used_fallback"))
                direct_models[arm][str(row.get("model"))] += 1
                if row.get("model") != "Qwen3-8B":
                    issue("unexpected_direct_opponent_model", match)
                if row.get("error_type"):
                    direct_errors[arm][str(row["error_type"])] += 1
                add_time(arm, "direct_model_call", row.get("elapsed_seconds"), match)
        if any(type(player) is not int or player not in range(7) or count != 1
               for player, count in seat_files.items()):
            issue("duplicate_or_invalid_worker_seat_log", match)
        if set(seat_files) != set(range(7)):
            missing.append(dict(match_hash=match, reason="missing_seat_worker_logs"))
        if not direct_call_count:
            missing.append(dict(match_hash=match, reason="missing_direct_opponent_model_calls"))
        if not focal_files:
            missing.append(dict(match_hash=match, reason="missing_focal_worker_log"))
            continue
        if focal_files != 1 or len(assignment_rows) != 1:
            issue("duplicate_or_missing_focal_assignment_log", match)
        else:
            assignment = assignment_rows[0]
            expected = {"seat": seat, "arm": arm, "mode": mode, "evidence_guard": guard,
                        "checkpoint_id": checkpoint, "proposer": "ids", "assignment_version": "wolf-policy-v1",
                        "assignment_seed": summary.get("seed"), "opponent_pool": POOL,
                        "competition_factory": True}
            if any(assignment.get(key) != value for key, value in expected.items()):
                issue("focal_assignment_profile_mismatch", match)
            if not focal_rows or focal_rows[0].get("event") != "policy_assignment":
                issue("assignment_not_first_focal_event", match)
        terminals = [row for row in focal_rows if row.get("event") == "game_result"]
        if len(terminals) > 1:
            issue("duplicate_focal_terminal", match)
        if resolved and (len(terminals) != 1 or any(terminals[0].get(key) != game.get(field) for key, field in
                                                  (("score", "score"), ("role_for_eval_only", "role"),
                                                   ("termination_reason", "termination_reason")))):
            issue("summary_terminal_does_not_match_focal_log", match)
        choices = [row for row in focal_rows if row.get("event") == "hybrid_choice"]
        if len(choices) != game.get("decisions", 0):
            issue("summary_choice_count_mismatch", match)
        ids = [row.get("decision_id") for row in choices]
        if None in ids or len(set(ids)) != len(ids):
            issue("duplicate_or_missing_choice_decision_id", match)
        deliveries = Counter(row.get("decision_id") for row in focal_rows
                             if row.get("event") == "hybrid_delivery" and row.get("accepted") is True)
        for row in focal_rows:
            if row.get("error_code"):
                counters[arm]["focal_delivery_error_events"] += 1
        for row in choices:
            count = counters[arm]
            count["focal_hybrid_choices"] += 1
            count["focal_model_valid"] += bool(row.get("model_valid"))
            count["focal_model_not_valid_including_forced"] += not bool(row.get("model_valid"))
            count["focal_used_fallback"] += bool(row.get("used_fallback"))
            count["focal_sampled"] += row.get("sampled") is True
            count["focal_non_sampled"] += row.get("sampled") is not True
            count["focal_forced_singleton_choices"] += (not row.get("model_valid")
                                                         and len(row.get("candidate_ids", [])) == 1)
            count["focal_delivery_accepted_exactly_once"] += deliveries[row.get("decision_id")] == 1
            count["focal_delivery_unconfirmed_or_duplicate"] += deliveries[row.get("decision_id")] != 1
            if row.get("selector_error"):
                selector_errors[arm][str(row["selector_error"])] += 1
            if row.get("model_error"):
                proposer_errors[arm][str(row["model_error"])] += 1
            expected_sampled = (mode == "hybrid_collect" and row.get("model_valid") is True
                                and len(row.get("candidate_ids", [])) > 1 and not row.get("selector_error"))
            if row.get("sampled") != bool(expected_sampled):
                issue("sampled_flag_mismatch", match)
            mass, selected = row.get("probabilities", []), row.get("choice_index")
            if (not isinstance(mass, list) or len(mass) != len(row.get("candidate_ids", []))
                    or type(selected) is not int or not 0 <= selected < len(mass)
                    or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in mass)
                    or not math.isclose(sum(mass), 1., abs_tol=1e-9)
                    or (expected_sampled and any(p <= 0 for p in mass))
                    or (not expected_sampled and mass != [float(index == selected) for index in range(len(mass))])):
                issue("logged_actual_selection_mass_invalid", match)
            if row.get("mode") != mode or row.get("checkpoint_id") != checkpoint:
                issue("choice_mode_or_checkpoint_mismatch", match)
            if row.get("opponent_pool") != POOL:
                issue("choice_opponent_roster_mismatch", match)
            config = row.get("proposer_configuration", {})
            config_counts[arm][json.dumps(config, sort_keys=True)] += 1
            generation_counts[arm][str(row.get("generator_id"))] += 1
            try:
                valid = (config["proposer"] == "ids" and config["prompt_version"] == "legacy-v1"
                         and config["device"] == "cpu" and config["requested_dtype"] == "bfloat16"
                         and config["requested_revision"] == QWEN_REVISION and config["actual_revision"] == QWEN_REVISION
                         and config["actual_dtype"] in {"unloaded", "bfloat16"}
                         and type(config.get("requested_cpu_threads")) is int and config["requested_cpu_threads"] == 4
                         and config.get("actual_cpu_threads") in {"unloaded", 4}
                         and (config["actual_dtype"] != "bfloat16" or config["actual_cpu_threads"] == 4)
                         and config.get("evidence_guard_version", "none") == guard
                         and "aggregation_version" not in config
                         and generator(config, guard != "none") == row.get("generator_id")
                         and generator(config, True, warm=True) == artifact["generator_id"])
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                issue("choice_generator_or_configuration_mismatch", match)
            if row.get("model_valid") and (config.get("actual_dtype") != "bfloat16"
                                            or config.get("actual_cpu_threads") != 4):
                issue("valid_model_proposal_has_unloaded_precision_or_wrong_threads", match)
            actual_guard = row.get("candidate_guard", {})
            if actual_guard and (guard == "none" or actual_guard.get("version") != GUARD):
                issue("candidate_guard_in_wrong_profile", match)
            if (actual_guard.get("reason") == "avoid_confirmed_innocent_when_unknown_available"
                    and actual_guard.get("before_target_ids") != actual_guard.get("retained_ids")):
                count["verified_innocent_guard_actual_interventions"] += 1
                if guard == "none" or actual_guard.get("version") != GUARD:
                    issue("guard_intervention_in_wrong_profile", match)
            add_time(arm, "focal_proposer", row.get("model_seconds"), match)
            if row.get("model_valid"):
                add_time(arm, "focal_valid_proposer_call", row.get("model_seconds"), match)
            if "selector_seconds" in row:
                add_time(arm, "focal_selector", row["selector_seconds"], match)

    for arm in PROFILES:
        if resolved_counts[arm] < 12:
            missing.append(dict(arm=arm, reason="fewer_than_12_resolved_games", resolved=resolved_counts[arm]))
        expected_resolved = summary.get("arms", {}).get(arm, {}).get("resolved")
        if expected_resolved != resolved_counts[arm]:
            issue("summary_resolved_count_mismatch", arm=arm)
    clean = not issues
    complete = clean and not missing
    return {
        "audit_integrity_passed": clean, "cohort_complete": complete,
        "exit_code": 1 if not clean else 0 if complete else 2,
        "interpretation": "Frozen prospective F operations only; no fitting, arm selection or winner-based stopping.",
        "minimum_resolved_per_arm": 12, "all_assigned_games_must_resolve": True,
        "protocol_issues": issues, "incomplete_reasons": missing,
        "E_F_overlap_hashes": overlap, "assigned_games": len(games),
        "assigned_by_arm": dict(arm_counts), "resolved_by_arm": dict(resolved_counts),
        "expected_profiles": {arm: dict(mode=mode, guard=guard, checkpoint_id=checkpoint)
                              for arm, (mode, guard, checkpoint) in PROFILES.items()},
        "per_arm_operations": {arm: {
            "counts": dict(counters[arm]), "selector_error_counts": dict(selector_errors[arm]),
            "proposer_error_counts": dict(proposer_errors[arm]),
            "direct_error_type_counts": dict(direct_errors[arm]),
            "direct_model_name_counts": dict(direct_models[arm]),
            "latency": {key: latency(values) for key, values in timings[arm].items()},
            "generator_id_counts": dict(generation_counts[arm]),
            "configuration_counts": [{"configuration": json.loads(key), "count": count}
                                     for key, count in config_counts[arm].items()],
        } for arm in PROFILES},
        "summary_reference": {key: summary.get(key) for key in
                              ("seed", "assignment_version", "receipt_count", "arms",
                               "observed_model_configurations", "model_configurations_by_arm")},
        "limitations": ["model_valid=false includes forced single-option choices and is not synonymous with fallback.",
                        "Direct model_call events identify validity, latency and model name, but not their own dtype/revision.",
                        "Median/p95/max concern recorded calls, including cold calls, not unobserved end-to-end response time.",
                        "Operational failures remain in their assigned cohorts; they do not filter games."],
        "provenance": {"helper_sha256": sha(Path(__file__)), "summary_sha256": sha(summary_path),
                       "training_artifact_sha256": sha(artifact_path), "training_tensor_sha256": tensor_hash,
                       "log_dir": str(log_dir.resolve()), "log_file_sha256_manifest": manifest},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.log_dir.is_dir():
        parser.error("LOG_DIR must be the actual F decisions directory, excluding runtime fixtures")
    if args.out.exists() or args.out.resolve() in {args.summary.resolve(), args.training_artifact.resolve()}:
        parser.error("Choose a new audit output path")
    try:
        report = analyze(args.log_dir, args.summary, args.training_artifact)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report = {"audit_integrity_passed": False, "cohort_complete": False, "exit_code": 1,
                  "protocol_issues": [{"error": "unreadable_or_structurally_invalid_audit_input",
                                       "exception_type": type(exc).__name__}],
                  "assigned_by_arm": {}, "resolved_by_arm": {},
                  "interpretation": "Audit could not finish; no model or cohort selection was performed."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    status = {key: report[key] for key in ("audit_integrity_passed", "cohort_complete", "exit_code")}
    status.update(protocol_issue_count=len(report.get("protocol_issues", [])),
                  incomplete_reason_count=len(report.get("incomplete_reasons", [])))
    print(json.dumps(status))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
