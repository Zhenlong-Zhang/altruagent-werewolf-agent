"""Hoffman2 runtime integration checks; fake backend/encoder, no LLM load."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from werewolf_agent import contextual_entry as entry
from werewolf_agent.strategy_policy import StrategyPolicy


class EntryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = {"model_id": "fixture/pinned", "revision": "a" * 40, "backend_version": "strategy-native-generation-v1",
                     "dtype": "torch.bfloat16", "device": "cuda", "max_new_tokens": 2048,
                     "temperature": .6, "model_seconds": 30., "input_tokens": 8192}
        self.config = {"schema": "contextual-runtime-v1", "collection_id": "fresh-fixture", "protocol_hash": "b" * 64,
            "mode": "sample", "checkpoint_id": "c" * 64, "lineage": {"controller": "fixture-v2"},
            "source_manifest_sha256": "d" * 64, "sdk_manifest_sha256": "e" * 64, "seed": 97001,
            "generator_spec": self.spec, "paths": {key: str(self.root / key) for key in
                ("socket", "tokenizer", "encoder", "journal_dir", "checkpoint", "source_manifest", "sdk_manifest")}}
        self.config["lineage"] = {"rules": "altruagent-seven-fixed-roles-v1", "source_manifest": self.config["source_manifest_sha256"],
            "generator": entry.fingerprint(self.spec), "controller": "fixture-controller-v2", "realizer": "typed-canonical-v1",
            "encoder": entry.ENCODER_ID, "serializer": entry.SERIALIZER, "population": "fixture-population-v1"}
        self.path = self.root / "runtime.json"

    def write(self, config=None):
        self.path.write_text(json.dumps(self.config if config is None else config))
        return self.path

    def provenance(self, **changes):
        return {"model_id": self.spec["model_id"], "revision": self.spec["revision"], "backend_version": self.spec["backend_version"],
                "actual_dtype": self.spec["dtype"], "device": self.spec["device"], "max_new_tokens": 2048,
                "temperature": .6, "max_input_tokens": 8192, "input_tokens": 100, "thinking_enabled": True,
                "finish_reason": "eos", "seed": 1, **changes}

    def test_explicit_config_rejects_missing_defaults_and_bad_modes(self):
        self.assertEqual(entry.load_config(self.write()), self.config)
        for mutate in (lambda c: c.pop("mode"), lambda c: c.update(mode="auto"), lambda c: c.update(seed=True),
                       lambda c: c["generator_spec"].pop("temperature"), lambda c: c.update(checkpoint_id=None),
                       lambda c: c["generator_spec"].update(temperature=.7),
                       lambda c: c.update(source_manifest_sha256="f" * 64),
                       lambda c: c["lineage"].update(encoder="wrong-encoder"),
                       lambda c: c["lineage"].update(population="")):
            bad = copy.deepcopy(self.config)
            mutate(bad)
            with self.assertRaises(ValueError):
                entry.load_config(self.write(bad))

    def test_provenance_is_checked_before_model_answer_is_usable(self):
        response = {"text": "<answer>{}</answer>", "provenance": self.provenance()}
        backend = SimpleNamespace(complete=lambda *args: copy.deepcopy(response))
        pinned = entry.PinnedBackend(backend, lambda _: 100, self.spec)
        result = pinned.complete([], 2048, .6, 1, time.monotonic() + 20.)
        self.assertEqual(result["provenance"]["finish_reason"], "eos")
        for key, bad in (("revision", "f" * 40), ("actual_dtype", "torch.float32"), ("thinking_enabled", False), ("input_tokens", 101)):
            response["provenance"] = self.provenance(**{key: bad})
            with self.assertRaises(entry.GeneratorProvenanceError):
                pinned.complete([], 2048, .6, 1, time.monotonic() + 20.)

    def test_source_and_loaded_sdk_bytes_are_verified(self):
        import altruagent.runner
        repo = Path(entry.__file__).resolve().parents[1]
        source = Path(self.config["paths"]["source_manifest"])
        source.write_text(json.dumps({"files": {str(path.relative_to(repo)): entry.sha(path)
                                                for path in (repo / "werewolf_agent").rglob("*.py")}}))
        sdkroot = self.root / "overlay"
        (sdkroot / "altruagent").mkdir(parents=True)
        runner = sdkroot / "altruagent/runner.py"
        runner.write_bytes(Path(altruagent.runner.__file__).read_bytes())
        sdkmanifest = sdkroot / "strategy-overlay.json"
        sdkmanifest.write_text(json.dumps({"schema": "strategy-sdk-overlay-v2", "files": {
            "altruagent/runner.py": {"overlay_sha256": entry.sha(runner)}}}))
        self.config["paths"]["sdk_manifest"] = str(sdkmanifest)
        self.config["source_manifest_sha256"], self.config["sdk_manifest_sha256"] = entry.sha(source), entry.sha(sdkmanifest)
        self.config["lineage"]["source_manifest"] = self.config["source_manifest_sha256"]
        with patch.object(altruagent.runner, "__file__", str(runner)):
            entry.verify_sources(self.config)
            original = source.read_bytes()
            omitted = json.loads(original)
            omitted["files"].pop("werewolf_agent/vendor/slot_attention.py")
            source.write_text(json.dumps(omitted))
            self.config["source_manifest_sha256"] = entry.sha(source)
            with self.assertRaisesRegex(ValueError, "omits an active"):
                entry.verify_sources(self.config)
            source.write_bytes(original)
            self.config["source_manifest_sha256"] = entry.sha(source)
            runner.write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "SDK source changed"):
                entry.verify_sources(self.config)

    def test_factory_shares_weights_but_not_owner_selector(self):
        from werewolf_agent.contextual_selector import ContextualSelector
        network, encoder = object(), object()
        manifest = {"checkpoint_id": self.config["checkpoint_id"], "lineage": copy.deepcopy(self.config["lineage"])}
        def fake_selector(network, encoder, owner, checkpoint_id, **kwargs):
            return SimpleNamespace(network=network, encoder=encoder, owner=owner, checkpoint_id=checkpoint_id, **kwargs)
        with patch.object(entry, "verify_sources"), patch.object(entry, "GenerationClient", return_value=object()), \
             patch.object(entry, "LocalTokenCounter", return_value=lambda _: 100), \
             patch("werewolf_agent.contextual_checkpoint.load_checkpoint", return_value=(network, manifest)) as loader, \
             patch("werewolf_agent.semantic_selector.FrozenEncoder", return_value=encoder) as enc, \
             patch("werewolf_agent.contextual_selector.ContextualSelector", side_effect=fake_selector):
            table = entry.create_agent(self.write())
            contexts = [SimpleNamespace(session_id="owned-factory", seat_position=i, agent_id=f"agent{i}") for i in (1, 5)]
            first, second = [table.strategy_transport_policy(context) for context in contexts]
            self.addCleanup(first.memory.close)
            self.addCleanup(second.memory.close)
            self.assertIs(first.selector.network, second.selector.network)
            self.assertIs(first.selector.encoder, second.selector.encoder)
            self.assertIsNot(first.selector, second.selector)
            self.assertNotEqual(first.selector.owner, second.selector.owner)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(enc.call_count, 1)
            rows = [json.loads(line) for line in first.memory.path.read_text().splitlines()]
            self.assertEqual(rows[0]["event"], "strategy_runtime")
            self.assertTrue(rows[0]["source_verified"])
            population_change = copy.deepcopy(self.config)
            population_change["lineage"]["population"] = "fresh-declared-population"
            updated = entry.create_agent(self.write(population_change))
            self.assertIsNotNone(updated.selector_factory)
            changed_controller = copy.deepcopy(population_change)
            changed_controller["lineage"]["controller"] = "changed-controller"
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                entry.create_agent(self.write(changed_controller))
            first.memory.close()
            other = entry.create_agent(self.write(dict(self.config, collection_id="stale-or-other")))
            with self.assertRaisesRegex(ValueError, "different runtime identity"):
                other.strategy_transport_policy(contexts[0])

    def test_teacher_factory_does_not_load_checkpoint_or_encoder(self):
        config = copy.deepcopy(self.config)
        config.update(mode="teacher", checkpoint_id=None)
        config["paths"].update(checkpoint=None, encoder=None)
        with patch.object(entry, "verify_sources"), patch.object(entry, "GenerationClient", return_value=object()), \
             patch.object(entry, "LocalTokenCounter", return_value=lambda _: 100), \
             patch("werewolf_agent.contextual_checkpoint.load_checkpoint") as loader, \
             patch("werewolf_agent.semantic_selector.FrozenEncoder") as encoder:
            table = entry.create_agent(self.write(config))
            self.assertIsNone(table.selector_factory)
            loader.assert_not_called()
            encoder.assert_not_called()

    def test_restart_sampler_uses_persisted_decision_identity(self):
        from werewolf_agent.contextual_selector import ContextualSelector
        owner = {"session_hash": "a" * 16, "seat": 6}
        network = SimpleNamespace(eval=lambda: network)
        continuous = ContextualSelector(network, object(), owner, "checkpoint", seed=12)
        restarted = ContextualSelector(network, object(), owner, "checkpoint", seed=12)
        snapshot = {"candidate_mask": [True, True], "owner": owner}
        with patch("werewolf_agent.contextual_selector.serialize", return_value=snapshot), \
             patch.object(continuous, "evaluate", return_value=([0., 0.], [.5, .5], 0.)), \
             patch.object(restarted, "evaluate", return_value=([0., 0.], [.5, .5], 0.)):
            continuous.choose({"owner": owner}, [], time.monotonic() + 10., decision_id=7)
            expected = continuous.choose({"owner": owner}, [], time.monotonic() + 10., decision_id=8)
            actual = restarted.choose({"owner": owner}, [], time.monotonic() + 10., decision_id=8)
            next_request = restarted.choose({"owner": owner}, [], time.monotonic() + 10., decision_id=9)
        self.assertEqual(expected["metadata"]["sampler_uniform"], actual["metadata"]["sampler_uniform"])
        self.assertEqual(expected["index"], actual["index"])
        self.assertNotEqual(actual["metadata"]["sampler_uniform"], next_request["metadata"]["sampler_uniform"])

    def test_terminal_audit_uses_only_owned_official_result_and_valid_returns(self):
        context = SimpleNamespace(session_id="terminal-owned", seat_position=6, agent_id="own-agent")
        for index, (final, returns, session, expected_source, verified) in enumerate([
            ({"your_return": 1}, {}, context.session_id, "final_result.your_return", True),
            ({"status": "completed"}, {"own-agent": -1, "other-agent": 1}, context.session_id, "state.returns[context.agent_id]", True),
            (None, {"own-agent": 1}, context.session_id, "state.returns[context.agent_id]", False),
            ({"your_return": True}, {}, context.session_id, "final_result.your_return", False),
            ({"your_return": 1}, {}, "other-session", "final_result.your_return", False),
        ]):
            path = self.root / f"terminal{index}.jsonl"
            policy = StrategyPolicy(context, object(), lambda _: 100, path)
            state = SimpleNamespace(final_result=final, returns=returns, session_id=session,
                                    is_terminal=True, status="completed", state_version=9, termination_reason="fixture")
            policy.on_game_result(state, context)
            row = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(row["terminal_audit"]["verified"], verified)
            self.assertEqual(row["terminal_audit"]["own_return_source"], expected_source)
            self.assertEqual(row["score"], row["terminal_audit"]["own_score"])


if __name__ == "__main__":
    unittest.main()
