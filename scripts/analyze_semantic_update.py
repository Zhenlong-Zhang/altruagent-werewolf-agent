"""Read-only semantic-PPO audit on its frozen E training slates, never F.

Distribution changes below are in-sample diagnostics, not counterfactual
game outcomes or an estimate of held-out playing strength. Run on Hoffman2.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform

from scripts.train_hybrid import episode
from werewolf_agent.selector import FEATURES, Selector, probabilities
from werewolf_agent.semantic_selector import SemanticSelector, file_sha256


def summarize(rows):
    count = len(rows)
    return {
        "sampled_training_slates": count,
        "mean_kl_logged_old_to_nn": sum(row["kl"] for row in rows) / count if count else None,
        "max_kl_logged_old_to_nn": max((row["kl"] for row in rows), default=None),
        "mean_total_variation": sum(row["tv"] for row in rows) / count if count else None,
        "max_total_variation": max((row["tv"] for row in rows), default=None),
        "greedy_argmax_changed_count": sum(row["old_argmax"] != row["nn_argmax"] for row in rows),
        "greedy_argmax_changed_fraction": (sum(row["old_argmax"] != row["nn_argmax"] for row in rows) / count
                                           if count else None),
    }


def census_hashes(path):
    receipts = json.loads(path.read_text())
    if not isinstance(receipts, list):
        raise ValueError("E census must be the frozen list of controller creation receipts")
    hashes = []
    for receipt in receipts:
        session = receipt.get("game_session_id") or receipt.get("match_id")
        if not isinstance(session, str) or not session:
            raise ValueError("E receipt is missing its game session identifier")
        hashes.append(hashlib.sha256(session.encode()).hexdigest()[:16])
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate match in E controller census")
    return set(hashes)


def audit(log_dir, start_path, artifact_path, encoder_path, census_path, training_report_path=None):
    old = Selector("hybrid_collect", start_path)
    # Relocation is explicit; revision and recorded staged-file hashes remain
    # checked by the unchanged semantic artifact constructor.
    previous_encoder = os.environ.get("WEREWOLF_SEMANTIC_ENCODER")
    os.environ["WEREWOLF_SEMANTIC_ENCODER"] = str(encoder_path.resolve())
    try:
        trained = SemanticSelector("hybrid_collect", artifact_path)
    finally:
        if previous_encoder is None:
            os.environ.pop("WEREWOLF_SEMANTIC_ENCODER", None)
        else:
            os.environ["WEREWOLF_SEMANTIC_ENCODER"] = previous_encoder
    record = trained.record
    recorded = record.get("train_match_hashes")
    if (not isinstance(recorded, list) or any(not isinstance(value, str) or not value for value in recorded)
            or len(recorded) != len(set(recorded))):
        raise ValueError("Artifact must record unique E training match hashes")
    generator = record.get("generator_id")
    if not isinstance(generator, str) or not generator:
        raise ValueError("Artifact must record its E collection generator")
    census = census_hashes(census_path)
    expected_set = set(recorded)
    games, sources, exclusions = {}, {}, Counter()
    for path in sorted(log_dir.rglob("*.jsonl")):
        item, reason = episode(path, old, generator)
        if reason:
            exclusions[reason] += 1
            continue
        terminal, _ = item
        match = terminal["match_hash"]
        if match in games:
            raise ValueError("More than one eligible sampled focal log in an E match")
        games[match] = item
        sources[match] = {"path": str(path.relative_to(log_dir)), "sha256": file_sha256(path)}
    checks = {
        "feature_schema_matches": tuple(record["features"]) == FEATURES,
        "parent_checkpoint_matches": record.get("parent") == old.checkpoint_id,
        "parent_sha256_matches": record.get("parent_sha256") == file_sha256(start_path),
        "eligible_match_set_matches_artifact": set(games) == expected_set,
        "census_match_set_matches_artifact": census == expected_set,
        "eligible_match_count_matches": len(games) == record.get("train_games") == len(recorded),
    }
    training_report = json.loads(training_report_path.read_text()) if training_report_path else None
    if training_report is not None:
        checks["training_report_artifact_matches"] = training_report.get("artifact") == record
        recorded_sources = training_report.get("train_log_sources", {})
        checks["training_source_match_set_matches"] = set(recorded_sources) == set(sources)
        checks["training_log_sha256_matches"] = all(
            recorded_sources.get(match, {}).get("sha256") == source["sha256"]
            for match, source in sources.items())
        checks["training_source_relative_paths_match"] = all(
            recorded_sources.get(match, {}).get("path") == source["path"]
            for match, source in sources.items())
    else:
        recorded_sources = {}

    # Reject a wrong/mutated dataset before running even an in-sample neural
    # evaluation. A failed report still records the discrepancies for review.
    if not games or not all(checks.values()):
        return {"audit_passed": False, "checks": checks,
                "reason": "E input provenance failed before neural evaluation",
                "eligible_matches": sorted(games), "artifact_matches": sorted(expected_set),
                "census_matches": sorted(census), "sources": sources,
                "training_log_sha_reference_available": training_report is not None}

    import torch
    trained._load()
    network = trained._network
    parameter_groups = {"actor_including_base": {}, "neural_actor_excluding_base": {}, "critic": {}}
    named_norms = {}
    for name, parameter in network.named_parameters():
        group = "critic" if name.startswith("critic.") else "actor_including_base"
        parameter_groups[group][name] = parameter.numel()
        if group == "actor_including_base" and name != "base_weights":
            parameter_groups["neural_actor_excluding_base"][name] = parameter.numel()
        named_norms[name] = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                             "l2_norm": float(parameter.detach().double().norm())}
    actor_initial = torch.zeros_like(network.actor[-1].weight)
    actor_head_changed = not torch.equal(network.actor[-1].weight.detach(), actor_initial)
    actor_head_bias_changed = bool((network.actor[-1].bias.detach() != 0).any())
    by_role, by_kind, metrics = defaultdict(list), defaultdict(list), []
    roles, scores, roles_and_scores = Counter(), Counter(), Counter()
    terminal_records = []
    initial_probability_max_error = 0.
    for match, (terminal, choices) in games.items():
        role, score = terminal["role_for_eval_only"], terminal["score"]
        roles[role] += 1
        scores[str(score)] += 1
        roles_and_scores[f"{role}:{score}"] += 1
        terminal_records.append({"match_hash": match, "seat": terminal["seat"],
                                 "role": role, "observed_training_score": score})
        for choice in choices:
            # Terminal role/score remain reporting labels, never policy inputs.
            features, labels = choice["features"], choice.get("candidate_labels")
            if not isinstance(labels, list) or len(labels) != len(features):
                raise ValueError("E choice is missing its exact before-action candidate labels")
            context = {"day": choice["day"], "phase": choice["phase"]}
            before = choice["probabilities"]
            reproduced_old = probabilities(features, old.weights)
            initial_probability_max_error = max(initial_probability_max_error,
                                                max(abs(a - b) for a, b in zip(before, reproduced_old)))
            after = trained.probabilities(features, labels, context)
            if len(after) != len(before) or any(not math.isfinite(value) or value <= 0 for value in after):
                raise ValueError("Invalid NN probability on an E training slate")
            metric = {
                "match_hash": match, "decision_id": choice["decision_id"],
                "kind": "speech" if choice["phase"] == "speech" else "action",
                "kl": sum(p * (math.log(p) - math.log(q)) for p, q in zip(before, after)),
                "tv": .5 * sum(abs(p - q) for p, q in zip(before, after)),
                "old_argmax": max(range(len(before)), key=before.__getitem__),
                "nn_argmax": max(range(len(after)), key=after.__getitem__),
                "old_logged_probabilities": before, "nn_probabilities": after,
            }
            metrics.append(metric)
            by_role[role].append(metric)
            by_kind[metric["kind"]].append(metric)
    summary = summarize(metrics)
    checks["training_roles_match_artifact"] = dict(roles) == record.get("train_roles")
    checks["sampled_choice_count_matches_artifact"] = len(metrics) == record.get("train_sampled_choices")
    checks["all_actor_and_critic_parameters_trainable"] = all(p.requires_grad for p in network.parameters())
    encoder_model = trained._encoder._runtime[1]
    checks["encoder_frozen_eval"] = not encoder_model.training and not any(p.requires_grad for p in encoder_model.parameters())
    comparisons = {}
    if training_report is not None:
        for key, summary_key in (("final_mean_kl", "mean_kl_logged_old_to_nn"),
                                 ("final_max_kl", "max_kl_logged_old_to_nn")):
            reference, actual = training_report[key], summary[summary_key]
            matched = math.isclose(reference, actual, rel_tol=1e-6, abs_tol=1e-8)
            checks[key + "_matches_training_report"] = matched
            comparisons[key] = {"recorded": reference, "recomputed": actual,
                                "absolute_error": abs(reference - actual),
                                "rtol": 1e-6, "atol": 1e-8}
    delta = [new - start for start, new in zip(old.weights, trained.weights)]
    names = [feature if condition == "intercept" else f"{feature} * {condition}"
             for condition in ("intercept", *FEATURES[4:]) for feature in FEATURES[:4]]
    return {
        "analysis": "semantic_saved_actor_in_sample_training_slate_audit",
        "interpretation": "E training-slate distribution changes only; no F inputs or counterfactual win claims.",
        "metric_weighting": "Equal weight per acknowledged sampled E slate; stable first-index argmax ties.",
        "audit_passed": all(checks.values()), "checks": checks,
        "inputs": {"log_dir": str(log_dir), "start": str(start_path), "start_sha256": file_sha256(start_path),
                   "artifact": str(artifact_path), "artifact_sha256": file_sha256(artifact_path),
                   "tensor_sha256": file_sha256(trained.tensor_path),
                   "census_sha256": file_sha256(census_path),
                   "training_report_sha256": file_sha256(training_report_path) if training_report_path else None,
                   "audit_script_sha256": file_sha256(Path(__file__))},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "torch_intraop_threads": torch.get_num_threads()},
        "parent_checkpoint_id": old.checkpoint_id, "semantic_checkpoint_id": trained.checkpoint_id,
        "generator_id": generator, "architecture": record["architecture"],
        "encoder_identity": record["encoder"], "resolved_local_encoder_path": str(trained.encoder_path),
        "eligible_games": len(games), "eligible_hashes": sorted(games), "census_hashes": sorted(census),
        "eligible_log_sources": sources, "excluded_seat_logs": dict(exclusions),
        "training_log_sha_reference_available": training_report is not None,
        "observed_training_roles": dict(roles), "observed_training_scores": dict(scores),
        "observed_training_roles_and_scores": dict(roles_and_scores), "training_terminals": terminal_records,
        "old_probability_reproduction_max_abs_error": initial_probability_max_error,
        "parameter_counts": {key: sum(values.values()) for key, values in parameter_groups.items()},
        "parameter_counts_by_name": parameter_groups,
        "total_actor_critic_parameters": sum(parameter.numel() for parameter in network.parameters()),
        "frozen_encoder_parameters": sum(parameter.numel() for parameter in encoder_model.parameters()),
        "neural_named_parameter_norms": named_norms,
        "zero_initialized_actor_head_changed": actor_head_changed or actor_head_bias_changed,
        "base_20_change": {"l2": math.sqrt(sum(value * value for value in delta)),
                           "max_abs": max(map(abs, delta)),
                           "changed_at_1e_minus_12": sum(abs(value) > 1e-12 for value in delta),
                           "coefficients": [dict(name=name, old=start, trained=new, delta=change)
                                            for name, start, new, change in zip(names, old.weights, trained.weights, delta)]},
        "in_sample_training_slate_metrics": summary,
        "in_sample_by_role": {role: summarize(rows) for role, rows in sorted(by_role.items())},
        "in_sample_by_kind": {kind: summarize(rows) for kind, rows in sorted(by_kind.items())},
        "training_report_kl_comparison": comparisons, "training_slate_diagnostics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.log_dir.is_dir():
        parser.error("LOG_DIR must be the existing frozen E training-log directory")
    protected = {path.resolve() for path in (args.start, args.artifact, args.census)}
    if args.training_report:
        protected.add(args.training_report.resolve())
    if args.out.resolve() in protected or args.out.exists():
        parser.error("Use a new report path; audit output cannot replace an input")
    result = audit(args.log_dir.resolve(), args.start, args.artifact, args.encoder_path,
                   args.census, args.training_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("audit_passed", "checks", "parameter_counts",
                                                  "in_sample_training_slate_metrics") if key in result}))
    if not result["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
