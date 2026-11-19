"""Hoffman2 semantic architecture checks; optional real local encoder/artifact probe."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import random
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from altruagent import DecisionContext, LegalAction
from scripts.train_semantic_ppo import episode_mean_sum, guarded_epoch, ppo_surrogate
from werewolf_agent.hybrid import HybridPolicy
from werewolf_agent.selector import FEATURES, INITIAL_WEIGHTS, design_row, probabilities
from werewolf_agent.semantic_selector import (
    ARCHITECTURE, ENCODER_NAME, ENCODER_REVISION, FrozenEncoder, SemanticSelector,
    build_network, context_inputs, file_sha256,
)


FEATURE_ROWS = [[.7134567891234567, -.3, .2, 0., 5 / 7, 0., 0., 1.],
                [.2, .4, 0., -1., 5 / 7, 0., 0., 1.],
                [.1, -.2, .8, 0., 5 / 7, 0., 0., 1.]]


def optimizers(model, base_lr=.01):
    actor = torch.optim.Adam([
        {"params": [model.base_weights], "lr": base_lr},
        {"params": [parameter for name, parameter in model.named_parameters()
                    if name != "base_weights" and not name.startswith("critic.")], "lr": 3e-4}])
    return actor, torch.optim.Adam(model.critic.parameters(), lr=1e-3)


def training_record(model, reward=1.):
    values, _ = context_inputs(FEATURE_ROWS, {"day": 2, "phase": "day_vote"})
    embeddings = torch.randn(3, 384)
    state_embedding = torch.randn(384)
    with torch.no_grad():
        logits, value = model(FEATURE_ROWS, embeddings, values, state_embedding)
        probs = torch.softmax(logits, dim=0).detach()
    return {"features": FEATURE_ROWS, "labels_embedding": embeddings,
            "context_vector": values, "context_embedding": state_embedding,
            "choice_index": 0, "old_probs": probs, "old_full_logp": probs.log(),
            "old_selected_logp": probs[0].log(), "return": torch.tensor(reward, dtype=torch.float64),
            "advantage": (torch.tensor(reward, dtype=torch.float64) - value.double()).detach()}


class SemanticChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(932)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_initial_logits_exact_legacy_and_context_not_rounded(self):
        model = build_network(INITIAL_WEIGHTS)
        values, _ = context_inputs(FEATURE_ROWS, {"day": 2, "phase": "day_vote"})
        logits, value = model(FEATURE_ROWS, torch.randn(3, 384), values, torch.randn(384))
        expected = torch.tensor([sum(x * w for x, w in zip(design_row(row), INITIAL_WEIGHTS))
                                 for row in FEATURE_ROWS], dtype=torch.float64)
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual(float(value), 0.)
        self.assertTrue(torch.allclose(torch.softmax(logits, dim=0),
                                      torch.tensor(probabilities(FEATURE_ROWS, INITIAL_WEIGHTS), dtype=torch.float64),
                                      rtol=1e-12, atol=1e-12))

    def test_attention_permutation_equivariance_after_nonzero_residual(self):
        model = build_network(INITIAL_WEIGHTS)
        with torch.no_grad():
            model.actor[-1].weight.fill_(.07)
        embeddings, state_embedding = torch.randn(3, 384), torch.randn(384)
        values, _ = context_inputs(FEATURE_ROWS, {"day": 2, "phase": "day_vote"})
        before, _ = model(FEATURE_ROWS, embeddings, values, state_embedding)
        order = [2, 0, 1]
        after, _ = model([FEATURE_ROWS[index] for index in order], embeddings[order], values, state_embedding)
        self.assertTrue(torch.allclose(after, before[order], rtol=1e-6, atol=1e-6))
        base = torch.tensor([sum(x * w for x, w in zip(design_row(row), INITIAL_WEIGHTS))
                             for row in FEATURE_ROWS], dtype=torch.float64)
        self.assertTrue(bool(((before - base).abs() <= 1.).all()))

    def test_context_ignores_future_metadata_and_canonicalizes_speech(self):
        speech = [row[:5] + [1.] + row[6:] for row in FEATURE_ROWS]
        a = context_inputs(speech, {"day": 2, "phase": "day_vote"})
        b = context_inputs(speech, {"day": 2, "phase": "speech", "terminal_score": -1,
                                    "winner": "wolves", "other_private_roles": {1: "wolf"}})
        self.assertEqual(a, b)
        self.assertEqual(len(a[0]), 8)
        self.assertNotIn("winner", a[1])

    def test_ppo_clips_both_advantage_signs(self):
        new_logp = torch.tensor([1.5, .5], dtype=torch.float64).log().requires_grad_()
        result = ppo_surrogate(new_logp, torch.zeros(2, dtype=torch.float64),
                               torch.tensor([1., -1.], dtype=torch.float64))
        self.assertTrue(torch.allclose(result, torch.tensor([1.2, -.8], dtype=torch.float64)))
        result.sum().backward()
        self.assertTrue(torch.equal(new_logp.grad, torch.zeros(2, dtype=torch.float64)))

    def test_episode_weighting_fixed_advantages_and_separate_critic(self):
        self.assertEqual(float(episode_mean_sum([torch.tensor(1.), torch.tensor(2.), torch.tensor(3.)], 2)), 3.)
        model = build_network(INITIAL_WEIGHTS)
        records = [training_record(model, reward=value) for value in (1., -1., -1.)]
        advantages = [record["advantage"].clone() for record in records]
        actor, critic = optimizers(model)
        outcome = guarded_epoch(torch, model, actor, critic, records, 2, mean_limit=100., max_limit=100.)
        self.assertTrue(outcome["accepted"])
        self.assertTrue(all(torch.equal(a, record["advantage"]) for a, record in zip(advantages, records)))
        self.assertEqual([float(record["advantage"]) for record in records], [1., -1., -1.])
        before = {name: value.clone() for name, value in model.state_dict().items() if not name.startswith("critic.")}
        critic.zero_grad(set_to_none=True)
        record = records[0]
        _, value = model(record["features"], record["labels_embedding"], record["context_vector"], record["context_embedding"])
        (value - 1).square().backward()
        critic.step()
        self.assertTrue(all(torch.equal(before[name], value) for name, value in model.state_dict().items()
                            if name in before))

    def test_kl_rejection_restores_full_model_and_populated_optimizer_states(self):
        model = build_network(INITIAL_WEIGHTS)
        records = [training_record(model)]
        actor, critic = optimizers(model)
        self.assertTrue(guarded_epoch(torch, model, actor, critic, records, 1,
                                      mean_limit=100., max_limit=100.)["accepted"])
        saved = copy.deepcopy((model.state_dict(), actor.state_dict(), critic.state_dict()))
        rejected = guarded_epoch(torch, model, actor, critic, records, 1, mean_limit=0., max_limit=0.)
        self.assertFalse(rejected["accepted"])
        self.assertGreater(rejected["attempted_mean_kl"], 0.)
        self.assertTrue(rejected["full_model_and_optimizers_restored"])
        self.assert_nested_equal(saved, (model.state_dict(), actor.state_dict(), critic.state_dict()))

    def test_artifact_validation_is_torch_free_and_checksum_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder"
            metadata = encoder / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            (metadata / "config.json.metadata").write_text(ENCODER_REVISION + "\n")
            (encoder / "config.json").write_text('{"hidden_size":384}')
            (encoder / "model.safetensors").touch()
            (encoder / "vocab.txt").touch()
            tensor = root / "selector.pt"
            torch.save(build_network(INITIAL_WEIGHTS).state_dict(), tensor)
            record = {"schema_version": 2, "features": FEATURES, "architecture": ARCHITECTURE,
                      "encoder": {"name": ENCODER_NAME, "revision": ENCODER_REVISION,
                                  "model_path": str(encoder)},
                      "base_weights": INITIAL_WEIGHTS, "weights_file": tensor.name,
                      "weights_sha256": file_sha256(tensor)}
            path = root / "selector.json"
            path.write_text(json.dumps(record))
            code = ("import sys; from werewolf_agent.semantic_selector import SemanticSelector; "
                    "p=SemanticSelector('hybrid_collect',sys.argv[1]); "
                    "assert 'torch' not in sys.modules; assert p.requires_semantic_context; "
                    "assert p._network is None")
            subprocess.run([sys.executable, "-c", code, str(path)], check=True)
            first = SemanticSelector("hybrid_collect", path)
            self.assertIsNone(first._network)
            tensor.write_bytes(tensor.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                SemanticSelector("hybrid_collect", path)

    def test_semantic_budget_rejects_after_embedding_and_final_host_conversion(self):
        for stage in ("embedding", "conversion"):
            with self.subTest(stage=stage):
                clock = [0.]
                selector = SemanticSelector.__new__(SemanticSelector)
                selector.mode, selector.rng = "hybrid_collect", random.Random(3)
                selector._network = build_network(INITIAL_WEIGHTS).eval()

                def encode(texts):
                    if stage == "embedding":
                        clock[0] = 6.
                    return torch.ones(len(texts), 384)

                selector._encoder = SimpleNamespace(encode=encode)
                original_tolist = torch.Tensor.tolist

                def synchronize(tensor):
                    result = original_tolist(tensor)
                    if stage == "conversion":
                        clock[0] = 6.
                    return result

                with patch("werewolf_agent.semantic_selector.time.monotonic", side_effect=lambda: clock[0]), \
                     patch.object(torch.Tensor, "tolist", new=synchronize):
                    with self.assertRaises(TimeoutError):
                        selector.choose(FEATURE_ROWS, ["a", "b", "c"],
                                        {"day": 2, "phase": "day_vote"}, deadline=5.)

    def test_normal_hybrid_semantic_success_and_failures_keep_legal_delivery_and_flags(self):
        for failure in (None, "runtime", "timeout", "late", "nan", "wrong_length", "bad_index"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory, \
                 patch.dict("os.environ", {"WEREWOLF_LOG_DIR": directory}, clear=True):
                clock, calls = [0.], []
                policy = HybridPolicy("llm_select", allow_tournament=True)
                policy.mode = "hybrid_collect"

                def choose(features, labels, context, *, deadline):
                    calls.append((features, labels, context, deadline))
                    if failure == "runtime":
                        raise RuntimeError("fixture failure")
                    if failure == "timeout":
                        raise TimeoutError("fixture timeout")
                    if failure == "late":
                        clock[0] = 6.
                    if failure == "nan":
                        return 0, [float("nan"), .25]
                    if failure == "wrong_length":
                        return 0, [1.]
                    if failure == "bad_index":
                        return 7, [.75, .25]
                    return 0, [.75, .25]

                policy.selector = SimpleNamespace(requires_semantic_context=True,
                                                  checkpoint_id="semantic-fixture", choose=choose)
                context = DecisionContext(session_id="semantic-normal-" + str(failure),
                                          tournament_id="fixture", game_type="werewolf",
                                          agent_id="fixture", seat_position=6)
                state = SimpleNamespace(observation="Your role: VILLAGER", new_messages=[],
                    raw={"players": [{"position": player, "name": f"Player{player}"} for player in range(7)],
                         "game_state": {"phase": "day_vote", "day": 2,
                                        "alive": [f"Player{player}" for player in range(7)],
                                        "dead": [], "vote_history": []}},
                    legal_actions=[LegalAction(action_id="1", label="Vote Player1", input={}),
                                   LegalAction(action_id="2", label="Vote Player2", input={})])

                def propose(view, options, speech):
                    slate = list(reversed(options))
                    return slate, [option.id for option in slate], None, .25

                with patch.object(policy, "_propose", side_effect=propose), \
                     patch("werewolf_agent.hybrid.time.monotonic", side_effect=lambda: clock[0]):
                    action = policy.choose_action(state, context)
                self.assertIn(action, state.legal_actions)
                self.assertEqual(action.action_id, "2" if failure is None else "1")
                policy.on_action_result({"accepted": True}, context)
                rows = [json.loads(line) for line in policy._logger.path.read_text().splitlines()]
                choice = next(row for row in rows if row["event"] == "hybrid_choice")
                delivery = next(row for row in rows if row["event"] == "hybrid_delivery")
                self.assertTrue(choice["model_valid"])
                self.assertEqual(choice["sampled"], failure is None)
                self.assertEqual(choice["used_fallback"], failure is not None)
                self.assertTrue(delivery["accepted"])
                self.assertEqual(choice["probabilities"], [.75, .25] if failure is None else [0., 1.])
                self.assertEqual(calls[0][1], ["Vote Player2", "Vote Player1"])
                self.assertEqual(calls[0][2], {"day": 2, "phase": "day_vote"})
                self.assertEqual(calls[0][3], 5.)
                if failure:
                    self.assertIn(choice["selector_error"], {"RuntimeError", "TimeoutError", "ValueError"})
                else:
                    self.assertIsNone(choice["selector_error"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SemanticChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = {"focused_checks_passed": result.testsRun}
    if args.encoder_path:
        encoder = FrozenEncoder(args.encoder_path)
        started = time.monotonic()
        first = encoder.encode(["I investigated Player2: wolf.", "I am uncertain who the wolf is."])
        report["encoder_cold_seconds"] = time.monotonic() - started
        started = time.monotonic()
        second = encoder.encode(["I investigated Player2: wolf.", "I am uncertain who the wolf is."])
        report["encoder_cached_seconds"] = time.monotonic() - started
        if not torch.equal(first, second) or not torch.allclose(first.norm(dim=1), torch.ones(2), atol=1e-5):
            raise ValueError("Real encoder reproducibility/normalization failure")
        tokenizer, model = encoder._runtime
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Real encoder must remain frozen/eval")
        report["encoder_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        report["encoder_frozen_eval"] = True
        del tokenizer, model, encoder
    if args.artifact:
        started = time.monotonic()
        selector = SemanticSelector("hybrid_collect", args.artifact, seed=12)
        report["artifact_validation_seconds"] = time.monotonic() - started
        labels = ["I investigated Player2: wolf.", "Who is your strongest suspect?", "I am uncertain."]
        started = time.monotonic()
        first = selector.probabilities(FEATURE_ROWS, labels, {"day": 2, "phase": "day_vote"})
        report["semantic_cold_seconds"] = time.monotonic() - started
        started = time.monotonic()
        second = selector.probabilities(FEATURE_ROWS, labels, {"day": 2, "phase": "day_vote"})
        report["semantic_cached_seconds"] = time.monotonic() - started
        if first != second:
            raise ValueError("Semantic checkpoint inference must be deterministic")
        report["checkpoint_id"] = selector.checkpoint_id
        report["probabilities"] = first
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
