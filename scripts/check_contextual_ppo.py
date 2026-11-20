"""Hoffman2-only synthetic learning/replay/rollback checks, no match claims."""

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from scripts.check_contextual_selector import FakeEncoder, fixture
from werewolf_agent.contextual_network import build_network
from werewolf_agent.contextual_checkpoint import load_checkpoint, save_checkpoint
from werewolf_agent.contextual_ppo import ContextualPPO
from werewolf_agent.contextual_selector import ContextualSelector, ENCODER_ID
from werewolf_agent.contextual_slots import SERIALIZER
from werewolf_agent.strategy_state import fingerprint


class PPOChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.network, self.encoder = build_network(), FakeEncoder()
        self.packet, self.candidates = fixture()
        self.lineage = {name: "synthetic-fixture" for name in
                        ("rules", "source_manifest", "generator", "controller", "realizer", "population")}
        self.lineage.update(encoder=ENCODER_ID, serializer=SERIALIZER)

    def episode(self, name="one", checkpoint="C0", singleton=False):
        packet, candidates = copy.deepcopy(self.packet), copy.deepcopy(self.candidates)
        if singleton:
            packet["legal_actions"] = packet["legal_actions"][:1]
            packet["packet_hash"] = fingerprint({key: value for key, value in packet.items() if key != "packet_hash"})
            candidates = [dict(candidates[0], packet_hash=packet["packet_hash"])]
        selector = ContextualSelector(self.network, self.encoder, packet["owner"], checkpoint)
        requests = [{"choice": selector.choose(packet, candidates, time.monotonic() + 30., decision_id=i),
                     "delivery_status": status, "executor_fallback": status != "accepted"}
                    for i, status in enumerate(("accepted", "rejected", "unknown"))]
        return {"game_id": name, "owner": packet["owner"], "checkpoint_id": checkpoint,
                "lineage": self.lineage, "complete": True, "terminal_return": 1.,
                "terminal_provenance": "synthetic-test-not-live-game", "requests": requests}

    def test_real_optimizer_steps_keep_requests_and_fixed_advantages(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage, config={"epochs": 2})
        episodes = [self.episode()]
        rows, _ = trainer._prepare(episodes)
        expected = [row["advantage"] for row in rows]
        report = trainer.update(episodes, chunk_size=2)
        self.assertGreater(report["accepted_steps"], 0)
        self.assertEqual(report["request_count"], 3)  # accepted/rejected/unknown all retained
        self.assertEqual(report["advantages"], expected)
        self.assertEqual(report["completed_games"], 1)
        self.assertTrue(trainer.critic_optimizer.state)
        self.assertTrue(trainer.actor_optimizer.state)
        fresh = self.episode("next", checkpoint="C1")
        self.assertNotEqual([row["choice"]["value"] for row in fresh["requests"]],
                            [row["choice"]["value"] for row in episodes[0]["requests"]])

    def assert_tree_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_trust_bound_restores_both_networks_and_optimizers(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage,
                                config={"epochs": 2, "learning_rate": .1, "mean_kl": 0., "maximum_kl": 0.})
        episodes = [self.episode()]
        before = trainer.transaction()
        report = trainer.update(episodes, chunk_size=3)
        self.assertEqual(report["accepted_steps"], 0)
        self.assertEqual(report["rejected_steps"], 1)
        self.assert_tree_equal(before, trainer.transaction())

    def test_infinite_kl_rejection_remains_serializable_and_restored(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage, config={"epochs": 1})
        episodes, before = [self.episode()], trainer.transaction()
        with patch("werewolf_agent.contextual_ppo.exact_categorical_kl",
                   side_effect=lambda old, new, legal: torch.full((len(old),), float("inf"))):
            report = trainer.update(episodes, chunk_size=3)
        self.assertEqual(report["accepted_steps"], 0)
        self.assertEqual(report["rejected_steps"], 1)
        json.dumps(report, allow_nan=False)
        self.assert_tree_equal(before, trainer.transaction())

    def test_nonfinite_step_rolls_back_populated_moments(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage, config={"epochs": 1})
        trainer.update([self.episode()], chunk_size=3)
        trainer.checkpoint_id = "C1"
        episodes = [self.episode("fresh", checkpoint="C1")]
        before = trainer.transaction()
        original_step = trainer.actor_optimizer.step
        def corrupt():
            original_step()
            with torch.no_grad():
                next(self.network.actor.parameters()).fill_(float("nan"))
        trainer.actor_optimizer.step = corrupt
        report = trainer.update(episodes, chunk_size=3)
        self.assertEqual(report["rejected_steps"], 1)
        self.assert_tree_equal(before, trainer.transaction())

    def test_singleton_after_learning_cannot_move_actor_via_adam_momentum(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage, config={"epochs": 1})
        trainer.update([self.episode()], chunk_size=3)
        trainer.checkpoint_id = "C1"
        before = copy.deepcopy(self.network.actor.state_dict())
        report = trainer.update([self.episode("forced", checkpoint="C1", singleton=True)], chunk_size=3)
        self.assertEqual(report["actor_requests"], 0)
        self.assert_tree_equal(before, self.network.actor.state_dict())

    def test_offpolicy_missing_terminal_and_duplicate_draws_reject(self):
        trainer = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage)
        valid = self.episode()
        mutations = (lambda ep: ep.update(complete=False),
                     lambda ep: ep.update(checkpoint_id="stale"),
                     lambda ep: ep["requests"][0]["choice"]["metadata"].update(mode="greedy"),
                     lambda ep: ep["requests"].append(copy.deepcopy(ep["requests"][0])),
                     lambda ep: ep.update(terminal_return=True))
        for mutate in mutations:
            episode = copy.deepcopy(valid)
            mutate(episode)
            with self.assertRaises(ValueError):
                trainer.update([episode], chunk_size=3)

    def test_singleton_only_game_stays_in_game_mean_denominator(self):
        normal, forced = self.episode(), self.episode("forced", singleton=True)
        other = copy.deepcopy(self.network)
        one = ContextualPPO(other, self.encoder, "C0", lineage=self.lineage, config={"epochs": 1})
        both = ContextualPPO(self.network, self.encoder, "C0", lineage=self.lineage, config={"epochs": 1})
        single_report = one.update([normal], chunk_size=3)
        mixed_report = both.update([normal, forced], chunk_size=3)
        self.assertAlmostEqual(single_report["updates"][0]["policy_loss"] / 2.,
                               mixed_report["updates"][0]["policy_loss"], places=6)

    def test_safe_tensor_checkpoint_roundtrip_and_tamper_reject(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "checkpoint"
            manifest = save_checkpoint(output, self.network, self.lineage, parent_id=None,
                                        training_report={"scope": "synthetic fixture, no live training"})
            loaded, reproduced = load_checkpoint(output)
            self.assertEqual(manifest, reproduced)
            self.assert_tree_equal(self.network.state_dict(), loaded.state_dict())
            with self.assertRaises(ValueError):
                save_checkpoint(output, self.network, self.lineage, parent_id=None, training_report={})
            (output / "training_report.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                load_checkpoint(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
