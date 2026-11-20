"""Verify fresh owner journals before BC/PPO; incomplete games stay accounted.

Run preparation on Slurm. Terminal/runtime audit fields are read only here,
never included in serialize(packet, candidates) or policy input tensors.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .contextual_slots import SERIALIZER, serialize
from .contextual_selector import ENCODER_ID
from .semantic_selector import file_sha256
from .strategy_state import fingerprint, generation_seed


def source_receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def validate_protocol(protocol):
    required = {"schema", "collection_id", "mode", "checkpoint_id", "lineage", "source_manifest_sha256",
                "sdk_manifest_sha256", "plan"}
    if (set(protocol) != required or protocol["schema"] != "contextual-collection-protocol-v1"
            or protocol["mode"] not in {"teacher", "sample", "greedy"}
            or not isinstance(protocol["collection_id"], str) or not protocol["collection_id"]
            or (protocol["checkpoint_id"] is None) != (protocol["mode"] == "teacher")
            or not isinstance(protocol["plan"], dict) or not protocol["plan"]):
        raise ValueError("Explicit precollection protocol and finite plan required")
    lineage = protocol["lineage"]
    names = {"rules", "source_manifest", "generator", "controller", "realizer", "encoder", "serializer", "population"}
    if (not isinstance(lineage, dict) or set(lineage) != names
            or any(not isinstance(value, str) or not value for value in lineage.values())
            or lineage["source_manifest"] != protocol["source_manifest_sha256"]
            or lineage["encoder"] != ENCODER_ID or lineage["serializer"] != SERIALIZER
            or lineage["rules"] != "altruagent-seven-fixed-roles-v1" or lineage["realizer"] != "typed-canonical-v1"):
        raise ValueError("Incomplete or inconsistent collection lineage")
    for value in (protocol["source_manifest_sha256"], protocol["sdk_manifest_sha256"],
                  *([protocol["checkpoint_id"]] if protocol["checkpoint_id"] is not None else [])):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
            raise ValueError("Collection identities must be full hashes")
    plan = protocol["plan"]
    if (any(type(plan.get(key)) is not int or plan[key] < 1 for key in ("maximum_created_games", "target_completed_games"))
            or plan["target_completed_games"] > plan["maximum_created_games"]):
        raise ValueError("Freeze finite game creation and completion quantities before collection")
    return fingerprint(protocol)


def same_generator(provenance, spec, seed):
    comparisons = {"model_id": "model_id", "revision": "revision", "backend_version": "backend_version",
                   "actual_dtype": "dtype", "device": "device", "max_new_tokens": "max_new_tokens",
                   "temperature": "temperature", "max_input_tokens": "input_tokens"}
    if (any(provenance.get(field) != spec[setting] for field, setting in comparisons.items())
            or provenance.get("thinking_enabled") is not True or provenance.get("seed") != seed
            or type(provenance.get("input_tokens")) is not int
            or not 1 <= provenance["input_tokens"] <= spec["input_tokens"]):
        raise ValueError("generator_provenance_mismatch")


def registration_owner(registration, protocol, protocol_hash):
    owner = {"session_hash": registration["session_hash"], "seat": registration["focal_seat"]}
    if (registration["collection_id"] != protocol["collection_id"] or registration["protocol_hash"] != protocol_hash
            or hashlib.sha256(registration["game_id"].encode()).hexdigest()[:16] != owner["session_hash"]
            or type(owner["seat"]) is not int or owner["seat"] not in range(7)):
        raise ValueError("registration_owner_protocol_mismatch")
    return owner


def verify_game(rows, registration, protocol, protocol_hash, runtime_config):
    """A registration exists before this owner's worker begins playing."""
    owner = registration_owner(registration, protocol, protocol_hash)
    if not rows or any(row.get("owner") != owner for row in rows):
        raise ValueError("missing_or_cross_owner_journal")
    runtime_rows = [(index, row) for index, row in enumerate(rows) if row["event"] == "strategy_runtime"]
    if not runtime_rows:
        raise ValueError("missing_predecision_runtime_proof")
    runtime_index, runtime = runtime_rows[0]
    runtime_fields = ("collection_id", "protocol_hash", "mode", "checkpoint_id", "lineage",
                      "source_manifest_sha256", "sdk_manifest_sha256", "generator_spec", "runtime_config_hash", "source_verified")
    if any({key: row.get(key) for key in runtime_fields} != {key: runtime.get(key) for key in runtime_fields}
           for _, row in runtime_rows[1:]):
        raise ValueError("runtime_changed_within_game")
    expected = {key: protocol[key] for key in ("collection_id", "mode", "checkpoint_id", "lineage",
                                              "source_manifest_sha256", "sdk_manifest_sha256")}
    expected["protocol_hash"] = protocol_hash
    if (any(runtime.get(key) != value for key, value in expected.items()) or runtime.get("source_verified") is not True
            or not isinstance(runtime.get("runtime_config_hash"), str)):
        raise ValueError("runtime_source_or_freshness_mismatch")
    if (fingerprint(runtime_config) != runtime["runtime_config_hash"]
            or any(runtime_config.get(key) != value for key, value in expected.items())
            or runtime_config.get("generator_spec") != runtime["generator_spec"]
            or protocol["lineage"]["generator"] != fingerprint(runtime["generator_spec"])
            or type(runtime_config.get("seed")) is not int):
        raise ValueError("archived_runtime_config_mismatch")
    terminals = [(index, row) for index, row in enumerate(rows) if row["event"] == "strategy_terminal"]
    if len(terminals) != 1:
        raise ValueError("missing_or_duplicate_terminal")
    terminal_index, terminal = terminals[0]
    audit, score = terminal.get("terminal_audit") or {}, terminal.get("score")
    if (audit.get("verified") is not True or audit.get("official_final_result_present") is not True
            or audit.get("is_terminal") is not True or audit.get("session_consistent") is not True
            or audit.get("own_return_source") not in {"final_result.your_return", "state.returns[context.agent_id]"}
            or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            or score not in {-1., 1.} or audit.get("own_score") != score
            or terminal.get("termination_reason") != "completed"):
        raise ValueError("unverified_faction_terminal")
    requests, demonstrations, seen_ids, role = [], [], set(), None
    counters = Counter()
    prepared_deliveries, delivery_status, action_status = {}, {}, {}
    for position, row in enumerate(rows):
        if row["event"] == "delivery_prepared":
            pending = row["pending"]
            prepared_deliveries.setdefault(pending["decision_id"], []).append((position, pending))
        elif row["event"] == "delivery_result":
            delivery_status[row["delivery_id"]] = row["status"]
        elif row["event"] == "delivery_reconciled":
            delivery_status[row["delivery_id"]] = "accepted"
        elif row["event"] == "action_result":
            action_status.setdefault(row.get("decision_id"), []).append(row.get("accepted") is True and not row.get("error"))
    last_decision = -1
    for position, row in enumerate(rows):
        if row["event"] != "strategy_request":
            continue
        if not runtime_index < position < terminal_index:
            raise ValueError("request_outside_runtime_terminal_bounds")
        identifier = row.get("decision_id")
        if type(identifier) is not int or identifier <= last_decision or identifier in seen_ids:
            raise ValueError("duplicate_or_unordered_decision")
        last_decision = identifier
        seen_ids.add(identifier)
        packet, candidates = row["packet"], row["candidates"]
        if packet["owner"] != owner or row.get("before_packet_hash") != packet["packet_hash"]:
            raise ValueError("request_owner_hash_mismatch")
        current_role = packet["identity"]["role"]
        if role is not None and current_role != role:
            raise ValueError("fixed_role_changed")
        role = current_role
        serialized = serialize(packet, candidates)
        counters["all_policy_requests"] += 1
        selection = row.get("behavior_selection")
        if protocol["mode"] == "sample":
            if selection is None:
                if row["behavior_kind"] in {"categorical", "greedy", "forced_singleton"}:
                    raise ValueError("sampled_request_missing_propensity")
                counters["deterministic_executor_requests"] += 1
                continue
            if (selection["checkpoint_id"] != protocol["checkpoint_id"]
                    or selection["metadata"]["mode"] != "sample"
                    or selection["metadata"]["decision_id"] != identifier
                    or row["requested_index"] != selection["index"]
                    or selection["metadata"]["serialized"] != serialized):
                raise ValueError("stale_or_changed_behavior_snapshot")
            same_generator(row["generator_provenance"], runtime["generator_spec"], generation_seed(runtime_config["seed"], owner, identifier))
            if selection["metadata"].get("selection_kind") == "sampled":
                seed_identity = hashlib.sha256(f"{runtime_config['seed']}:{fingerprint(owner)}".encode()).hexdigest()
                digest = hashlib.sha256(f"{seed_identity}:{identifier}".encode()).digest()
                expected_uniform = (int.from_bytes(digest[:8], "big") >> 11) / 2**53
                if (selection["metadata"].get("sampler") != "owner-decision-sha256-uniform53-v1"
                        or selection["metadata"].get("sampler_uniform") != expected_uniform):
                    raise ValueError("owner_decision_sampler_provenance_mismatch")
            requests.append({"choice": selection, "decision_id": identifier,
                             "executor_index": row["executor_index"], "executor_overridden": row["executor_overridden"],
                             "error_type": row["error_type"]})
            counters["retained_sampled_or_singleton_requests"] += 1
        elif protocol["mode"] == "teacher":
            if selection is not None:
                raise ValueError("teacher_collection_contains_learned_policy")
            if not row.get("bc_eligible"):
                counters["ineligible_teacher_requests"] += 1
                continue
            teacher = row.get("teacher_choice") or {}
            preferred = teacher.get("index")
            if (teacher.get("source") != "actual_native_completion_preference" or row["teacher_source"] != "actual_generation"
                    or type(preferred) is not int or not 0 <= preferred < len(candidates)
                    or row["requested_index"] != preferred or row["executor_index"] != preferred
                    or row["executor_overridden"] or row["error_type"] is not None
                    or not serialized["candidate_mask"][preferred]
                    or teacher.get("candidate_id") != candidates[preferred]["id"]):
                raise ValueError("fabricated_or_gated_teacher_target")
            same_generator(row["generator_provenance"], runtime["generator_spec"], generation_seed(runtime_config["seed"], owner, identifier))
            candidate = candidates[preferred]
            if candidate.get("delivery") == "action":
                confirmed = action_status.get(identifier) == [True]
            else:
                attempts = prepared_deliveries.get(identifier, [])
                confirmed = False
                if len(attempts) == 1:
                    delivery_position, pending = attempts[0]
                    confirmed = (position < delivery_position < terminal_index
                        and delivery_status.get(pending["delivery_id"]) == "accepted"
                        and pending["candidate_id"] == candidate["id"] and pending["packet_hash"] == packet["packet_hash"]
                        and pending["canonical_text"] == candidate["text"]
                        and pending["recipients"] == candidate["recipients"]
                        and pending["assertions"] == candidate["assertions"])
            if not confirmed:
                counters["teacher_execution_unconfirmed_or_rejected"] += 1
                continue
            demonstrations.append({"serialized": serialized, "teacher_index": preferred,
                                   "decision_id": identifier, "game_id": registration["game_id"]})
        else:
            if selection is not None:
                if (selection["checkpoint_id"] != protocol["checkpoint_id"]
                        or selection["metadata"]["mode"] != "greedy"
                        or selection["metadata"]["decision_id"] != identifier
                        or row["requested_index"] != selection["index"]
                        or selection["metadata"]["serialized"] != serialized
                        or selection["index"] != max(range(len(selection["probabilities"])),
                                                       key=lambda index: selection["probabilities"][index])):
                    raise ValueError("stale_or_changed_greedy_evaluation")
                same_generator(row["generator_provenance"], runtime["generator_spec"], generation_seed(runtime_config["seed"], owner, identifier))
            elif row["behavior_kind"] in {"categorical", "greedy", "forced_singleton"}:
                raise ValueError("greedy_evaluation_missing_behavior_proof")
            counters["greedy_evaluation_requests"] += 1
    return {"game_id": registration["game_id"], "owner": owner, "role": role or "unknown",
            "checkpoint_id": protocol["checkpoint_id"], "lineage": protocol["lineage"],
            "complete": True, "terminal_return": float(score),
            "terminal_provenance": {"event_index": terminal_index, "receipt_hash": fingerprint(terminal)},
            "requests": requests, "demonstrations": demonstrations, "accounting": dict(counters)}


def prepare_batch(protocol_path, registry_path):
    protocol_path, registry_path = Path(protocol_path), Path(registry_path)
    protocol = json.loads(protocol_path.read_text())
    protocol_hash = validate_protocol(protocol)
    registry = [json.loads(line) for line in registry_path.read_text().splitlines()]
    episodes, excluded, sources = [], [], [source_receipt(protocol_path), source_receipt(registry_path)]
    training_manifest = None
    seen_ids, seen_owners = set(), set()
    for registration in registry:
        registration_owner(registration, protocol, protocol_hash)
        identifier, owner_key = registration["game_id"], (registration["session_hash"], registration["focal_seat"])
        if identifier in seen_ids or owner_key in seen_owners:
            raise ValueError("Registry duplicates game/focal seat; never count it twice")
        seen_ids.add(identifier); seen_owners.add(owner_key)
        path = Path(registration["journal_path"])
        if not path.is_file():
            excluded.append({"game_id": identifier, "reason": "missing_journal"})
            continue
        config_path = Path(registration["runtime_config_path"])
        if not config_path.is_file():
            excluded.append({"game_id": identifier, "reason": "missing_archived_runtime_config"})
            continue
        receipt = source_receipt(path)
        sources.extend([receipt, source_receipt(config_path)])
        try:
            config = json.loads(config_path.read_text())
            for name, hash_name in (("source_manifest", "source_manifest_sha256"), ("sdk_manifest", "sdk_manifest_sha256")):
                manifest_receipt = source_receipt(config["paths"][name])
                if manifest_receipt["sha256"] != protocol[hash_name]:
                    raise ValueError("archived_source_or_sdk_manifest_mismatch")
                sources.append(manifest_receipt)
                if name == "source_manifest":
                    training_manifest = manifest_receipt
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            episode = verify_game(rows, registration, protocol, protocol_hash, config)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            excluded.append({"game_id": identifier, "reason": str(exc), "journal_sha256": receipt["sha256"]})
        else:
            episode["terminal_provenance"]["journal_sha256"] = receipt["sha256"]
            episodes.append(episode)
    return {"schema": "verified-contextual-episodes-v1", "collection_id": protocol["collection_id"],
            "mode": protocol["mode"], "protocol_hash": protocol_hash, "lineage": protocol["lineage"],
            "episodes": episodes, "excluded": excluded, "source_receipts": sources,
            "training_source_manifest": training_manifest,
            "accounting": {"registered_games": len(registry), "eligible_completed_games": len(episodes),
                           "creation_cap_exceeded": len(registry) > protocol["plan"]["maximum_created_games"],
                           "exclusion_reasons": dict(Counter(row["reason"] for row in excluded))}}


def verify_sources(prepared):
    receipts = prepared.get("source_receipts")
    if not isinstance(receipts, list) or len(receipts) < 2:
        raise ValueError("Missing raw collector provenance")
    for receipt in receipts:
        if file_sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Collector source changed after batch preparation")
    receipt = prepared.get("training_source_manifest")
    if not isinstance(receipt, dict) or receipt["sha256"] != prepared["lineage"]["source_manifest"]:
        raise ValueError("Missing pinned fitting source manifest")
    manifest = json.loads(Path(receipt["path"]).read_text())
    files = manifest.get("files")
    required = {f"werewolf_agent/{name}.py" for name in (
        "contextual_data", "contextual_slots", "contextual_selector", "contextual_network", "contextual_checkpoint",
        "contextual_ppo", "contextual_bc", "semantic_selector", "strategy_candidates", "strategy_candidates_v2",
        "strategy_state", "beliefs", "vendor/slot_attention", "vendor/event_ppo")}
    required.update(f"scripts/{name}.py" for name in ("train_contextual_ppo", "train_contextual_bc", "prepare_contextual_data"))
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Fitting source manifest omits active modules")
    root = Path(__file__).resolve().parents[1]
    for name, expected in files.items():
        relative = Path(name)
        target = root / relative
        if (relative.is_absolute() or ".." in relative.parts or target.is_symlink()
                or not target.resolve().is_relative_to(root) or file_sha256(target) != expected):
            raise ValueError("Fitting source differs from the declared collection lineage")
