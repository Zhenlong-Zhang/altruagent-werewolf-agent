"""Slurm-only fresh-journal/teacher/terminal fixtures; no live game claims."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from scripts.check_contextual_selector import FakeEncoder, fixture
from werewolf_agent.contextual_bc import warm_start
from werewolf_agent.contextual_data import prepare_batch, validate_protocol, verify_game, verify_sources
from werewolf_agent.contextual_network import build_network
from werewolf_agent.contextual_selector import ContextualSelector, ENCODER_ID
from werewolf_agent.contextual_slots import SERIALIZER
from werewolf_agent.strategy_state import fingerprint, generation_seed


def journal_fixture(mode="sample"):
    import torch
    torch.set_num_threads(1); torch.manual_seed(71)
    packet, candidates = fixture()
    owner = packet["owner"]
    network, encoder = build_network(), FakeEncoder()
    spec = {"model_id": "fixture/pinned", "revision": "a" * 40, "backend_version": "strategy-native-generation-v1",
            "dtype": "torch.bfloat16", "device": "cuda", "max_new_tokens": 2048,
            "temperature": .6, "model_seconds": 30., "input_tokens": 8192}
    lineage = {key: "fixture" for key in ("rules", "source_manifest", "generator", "controller", "realizer", "population")}
    lineage.update(encoder=ENCODER_ID, serializer=SERIALIZER, generator=fingerprint(spec), source_manifest="d" * 64,
                   rules="altruagent-seven-fixed-roles-v1", realizer="typed-canonical-v1")
    checkpoint = None if mode == "teacher" else "c" * 64
    protocol = {"schema": "contextual-collection-protocol-v1", "collection_id": "synthetic-fresh-batch",
                "mode": mode, "checkpoint_id": checkpoint, "lineage": lineage,
                "source_manifest_sha256": "d" * 64, "sdk_manifest_sha256": "e" * 64,
                "plan": {"maximum_created_games": 2, "target_completed_games": 2, "scope": "synthetic fixtures only"}}
    protocol_hash = validate_protocol(protocol)
    runtime_config = {key: protocol[key] for key in ("collection_id", "mode", "checkpoint_id", "lineage",
                                                   "source_manifest_sha256", "sdk_manifest_sha256")}
    runtime_config.update(protocol_hash=protocol_hash, generator_spec=spec, seed=97001)
    runtime = dict(runtime_config, event="strategy_runtime", owner=owner, source_verified=True,
                   runtime_config_hash=fingerprint(runtime_config))
    selector = ContextualSelector(network, encoder, owner, "c" * 64, seed=97001,
                                   mode="greedy" if mode == "greedy" else "sample")
    choice = selector.choose(packet, candidates, time.monotonic() + 30., decision_id=1)
    preferred = 1
    provenance = dict(model_id=spec["model_id"], revision=spec["revision"], backend_version=spec["backend_version"],
                      actual_dtype=spec["dtype"], device=spec["device"], max_new_tokens=spec["max_new_tokens"],
                      temperature=spec["temperature"], max_input_tokens=spec["input_tokens"], thinking_enabled=True)
    provenance.update(seed=generation_seed(97001, owner, 1), input_tokens=100)
    request = {"event": "strategy_request", "owner": owner, "decision_id": 1, "packet": packet, "candidates": candidates,
               "before_packet_hash": packet["packet_hash"], "behavior_selection": None if mode == "teacher" else choice,
               "behavior_kind": "teacher_demo" if mode == "teacher" else "greedy" if mode == "greedy" else "categorical",
               "requested_index": preferred if mode == "teacher" else choice["index"], "executor_index": preferred if mode == "teacher" else choice["index"],
               "executor_overridden": False, "error_type": None, "generator_provenance": provenance,
               "teacher_choice": {"index": preferred, "candidate_id": candidates[preferred]["id"], "source": "actual_native_completion_preference"},
               "teacher_source": "actual_generation", "bc_eligible": mode == "teacher"}
    terminal = {"event": "strategy_terminal", "owner": owner, "score": 1., "termination_reason": "completed",
                "terminal_audit": {"verified": True, "official_final_result_present": True, "is_terminal": True,
                                   "session_consistent": True, "own_return_source": "final_result.your_return", "own_score": 1.}}
    action_ack = {"event": "action_result", "owner": owner, "decision_id": 1, "accepted": True, "error": None}
    registration = {"game_id": "r04-diagnostic-verified_wolf_over_claim-0-False", "session_hash": owner["session_hash"],
                    "focal_seat": owner["seat"], "collection_id": protocol["collection_id"], "protocol_hash": protocol_hash}
    return [runtime, request, action_ack, terminal], registration, protocol, runtime_config, network, encoder


class DataChecks(unittest.TestCase):
    def test_greedy_evaluation_binds_actual_checkpoint_and_generator(self):
        rows, registration, protocol, config, _, _ = journal_fixture("greedy")
        result = verify_game(rows, registration, protocol, fingerprint(protocol), config)
        self.assertEqual(result["accounting"]["greedy_evaluation_requests"], 1)
        rows[1]["behavior_selection"]["checkpoint_id"] = "stale"
        with self.assertRaises(ValueError):
            verify_game(rows, registration, protocol, fingerprint(protocol), config)

    def test_request_rejection_unknown_keeps_original_sample(self):
        rows, registration, protocol, config, _, _ = journal_fixture()
        rows[1].update(executor_overridden=True, error_type="late_executor")
        rows[2].update(accepted=None, error="DELIVERY_UNKNOWN")
        result = verify_game(rows, registration, protocol, fingerprint(protocol), config)
        self.assertEqual(len(result["requests"]), 1)
        self.assertEqual(result["requests"][0]["choice"], rows[1]["behavior_selection"])
        self.assertEqual(result["terminal_return"], 1.)

    def test_teacher_label_requires_actual_preference_and_confirmed_execution(self):
        rows, registration, protocol, config, _, _ = journal_fixture("teacher")
        result = verify_game(rows, registration, protocol, fingerprint(protocol), config)
        self.assertEqual(result["demonstrations"][0]["teacher_index"], 1)
        rows[2]["accepted"] = None
        result = verify_game(rows, registration, protocol, fingerprint(protocol), config)
        self.assertEqual(result["demonstrations"], [])
        rows[1]["teacher_choice"]["index"] = 0
        with self.assertRaises(ValueError):
            verify_game(rows, registration, protocol, fingerprint(protocol), config)

    def test_missing_terminal_stale_runtime_and_changed_sampler_reject(self):
        rows, registration, protocol, config, _, _ = journal_fixture()
        with self.assertRaises(ValueError):
            verify_game(rows[:-1], registration, protocol, fingerprint(protocol), config)
        for mutate in (lambda records: records[0].update(collection_id="old-batch"),
                       lambda records: records[1]["behavior_selection"]["metadata"].update(sampler_uniform=.12345),
                       lambda records: records[-1].update(termination_reason="failed"),
                       lambda records: records[1].update(decision_id=True)):
            bad = copy.deepcopy(rows); mutate(bad)
            with self.assertRaises(ValueError):
                verify_game(bad, registration, protocol, fingerprint(protocol), config)

    def test_full_registry_accounts_missing_games_and_pins_raw_sources(self):
        rows, registration, protocol, config, _, _ = journal_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal, runtime = root / "private.jsonl", root / "runtime.json"
            project = Path(__file__).resolve().parents[1]
            source_manifest, sdk_manifest = root / "sources.json", root / "sdk.json"
            files = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for folder in (project / "werewolf_agent", project / "scripts")
                     for path in folder.rglob("*.py")}
            source_manifest.write_text(json.dumps({"files": files}))
            sdk_manifest.write_text(json.dumps({"schema": "synthetic-sdk-receipt"}))
            source_sha, sdk_sha = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (source_manifest, sdk_manifest))
            protocol.update(source_manifest_sha256=source_sha, sdk_manifest_sha256=sdk_sha)
            protocol["lineage"]["source_manifest"] = source_sha
            protocol_hash = fingerprint(protocol)
            config.update(protocol_hash=protocol_hash, source_manifest_sha256=source_sha, sdk_manifest_sha256=sdk_sha,
                          paths={"source_manifest": str(source_manifest), "sdk_manifest": str(sdk_manifest)})
            rows[0].update(protocol_hash=protocol_hash, source_manifest_sha256=source_sha, sdk_manifest_sha256=sdk_sha,
                           runtime_config_hash=fingerprint(config))
            registration["protocol_hash"] = protocol_hash
            journal.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            runtime.write_text(json.dumps(config))
            registration.update(journal_path=str(journal), runtime_config_path=str(runtime))
            missing = dict(registration, game_id="missing", session_hash=hashlib.sha256(b"missing").hexdigest()[:16], journal_path=str(root / "absent"))
            protocol_path, registry_path = root / "protocol.json", root / "registry.jsonl"
            protocol_path.write_text(json.dumps(protocol))
            registry_path.write_text(json.dumps(registration) + "\n" + json.dumps(missing) + "\n")
            result = prepare_batch(protocol_path, registry_path)
            self.assertEqual(result["accounting"]["registered_games"], 2)
            self.assertEqual(result["accounting"]["eligible_completed_games"], 1, result["excluded"])
            self.assertEqual(result["excluded"][0]["reason"], "missing_journal")
            verify_sources(result)
            runtime.write_text("{}")
            with self.assertRaises(ValueError):
                verify_sources(result)

    def test_bc_game_disjoint_fit_uses_actual_teacher_targets(self):
        rows, registration, protocol, config, network, encoder = journal_fixture("teacher")
        base = verify_game(rows, registration, protocol, fingerprint(protocol), config)
        episodes = []
        for index in range(4):
            item = copy.deepcopy(base); item["game_id"] = f"synthetic-bc-{index}"
            item["demonstrations"][0]["game_id"] = item["game_id"]
            episodes.append(item)
        report, optimizers = warm_start(network, encoder, episodes, epochs=2, learning_rate=.0003,
                                        dev_fraction=.25, seed=71, chunk_size=2)
        self.assertFalse(set(report["training_game_hashes"]) & set(report["development_game_hashes"]))
        self.assertEqual(len(report["history"]), 2)
        self.assertIn(report["selected_epoch"], (1, 2))
        self.assertTrue(optimizers["actor_optimizer"]["state"])
        self.assertTrue(optimizers["critic_optimizer"]["state"])
        bad = copy.deepcopy(episodes)
        bad[0]["demonstrations"][0]["teacher_index"] = True
        with self.assertRaises(ValueError):
            warm_start(network, encoder, bad, epochs=1, learning_rate=.0003, dev_fraction=.25, seed=71, chunk_size=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
