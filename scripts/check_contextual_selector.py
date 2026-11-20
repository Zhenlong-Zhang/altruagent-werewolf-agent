"""Hoffman2-only structured actor/critic and replay checks; fake text encoder."""

import copy
import hashlib
import json
import time
import unittest

import torch

from scripts.bench_strategy_models import scenarios
from werewolf_agent.contextual_network import build_network
from werewolf_agent.contextual_selector import ContextualSelector, tensorize
from werewolf_agent.contextual_slots import serialize
from werewolf_agent.strategy_state import fingerprint


class FakeEncoder:
    def encode(self, texts):
        values = []
        for text in texts:
            generator = torch.Generator().manual_seed(int(hashlib.sha256(text.encode()).hexdigest()[:12], 16))
            values.append(torch.nn.functional.normalize(torch.randn(384, generator=generator), dim=0))
        return torch.stack(values)


def fixture(shift=0):
    row = next(row for row in scenarios() if row["scenario"] == "verified_wolf_over_claim"
               and row["shift"] == shift and not row["order_reversed"])
    packet = copy.deepcopy(row["packet"])
    packet["schema_version"] = "strategy-observation-v2"
    packet["self_assertions"] = []
    packet["window"].update(owner=dict(packet["owner"], day=packet["identity"]["day"], window_kind="day_vote_discussion"),
                            private_accepted=0, stage="opening", terminated=False, transcript_error=None)
    packet["packet_hash"] = fingerprint({key: value for key, value in packet.items() if key != "packet_hash"})
    candidates = [dict(row, delivery="action", intent=row["kind"], schema_version="strategy-candidates-v2",
                       packet_hash=packet["packet_hash"], available=True, unavailable_reason=None)
                  for row in packet["legal_actions"]]
    return packet, candidates


class ContextualChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.network, self.encoder = build_network().eval(), FakeEncoder()
        self.packet, self.candidates = fixture()

    def test_snapshot_replay_and_actual_sample_distribution(self):
        selector = ContextualSelector(self.network, self.encoder, self.packet["owner"], "fixture")
        choice = selector.choose(self.packet, self.candidates, time.monotonic() + 20., decision_id=12)
        snapshot = choice["metadata"]["serialized"]
        self.assertEqual(json.loads(json.dumps(snapshot, allow_nan=False)), snapshot)
        _, probabilities, value = selector.evaluate(snapshot)
        torch.testing.assert_close(probabilities, torch.tensor(choice["probabilities"]), atol=1e-6, rtol=1e-5)
        self.assertEqual(float(value), choice["value"])
        self.assertEqual(choice["metadata"]["selection_kind"], "sampled")
        self.assertIsNotNone(choice["metadata"]["draw_id"])
        self.assertAlmostEqual(sum(choice["probabilities"]), 1., places=12)
        self.assertEqual(choice["metadata"]["decision_id"], 12)

    def test_critic_update_does_not_change_actor(self):
        inputs = tensorize([serialize(self.packet, self.candidates)], self.encoder)
        before = self.network.actor(inputs).detach().clone()
        optimizer = torch.optim.Adam(self.network.critic.parameters(), lr=.001)
        optimizer.zero_grad()
        self.network.critic(inputs).square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(self.network.actor(inputs), before, atol=0., rtol=0.)
        self.assertFalse({id(value) for value in self.network.actor.parameters()}
                         & {id(value) for value in self.network.critic.parameters()})

    def test_typed_seat_rotation_and_candidate_order(self):
        with torch.no_grad():
            self.network.actor.body.relation_weights.normal_(0., .1)
            self.network.critic.body.relation_weights.normal_(0., .1)
        first = tensorize([serialize(self.packet, self.candidates)], self.encoder)
        rotated_packet, rotated_candidates = fixture(1)
        second = tensorize([serialize(rotated_packet, rotated_candidates)], self.encoder)
        logits, values = self.network(first)
        rotated_logits, rotated_values = self.network(second)
        for index, candidate in enumerate(self.candidates):
            rotated_target = None if candidate["target"] is None else (candidate["target"] + 1) % 7
            corresponding = next(i for i, row in enumerate(rotated_candidates) if row["target"] == rotated_target)
            torch.testing.assert_close(logits[0, index], rotated_logits[0, corresponding], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(values, rotated_values, atol=1e-6, rtol=1e-5)
        reversed_inputs = tensorize([serialize(self.packet, list(reversed(self.candidates)))], self.encoder)
        torch.testing.assert_close(self.network.actor(reversed_inputs), logits.flip(1), atol=1e-6, rtol=1e-5)

    def test_changed_claim_is_visible_but_does_not_rewrite_fact(self):
        before = serialize(self.packet, self.candidates)
        modified = copy.deepcopy(self.packet)
        modified["attributed_claims"][0]["text"] = "Player1 is not a wolf; my earlier claim was uncertain."
        modified["packet_hash"] = fingerprint({key: value for key, value in modified.items() if key != "packet_hash"})
        candidates = [dict(row, packet_hash=modified["packet_hash"]) for row in self.candidates]
        after = serialize(modified, candidates)
        self.assertNotEqual(before["slots_hash"], after["slots_hash"])
        self.assertEqual([row for row in before["slots"] if row["type"] == 2],
                         [row for row in after["slots"] if row["type"] == 2])

    def test_owner_future_private_and_snapshot_modification_reject(self):
        selector = ContextualSelector(self.network, self.encoder, dict(self.packet["owner"], seat=6), "other")
        with self.assertRaises(ValueError):
            selector.choose(self.packet, self.candidates, time.monotonic() + 20.)
        for mutate in (lambda packet: packet["window"].update(winner="wolves"),
                       lambda packet: packet["attributed_claims"][0].update(channel="private", recipients=[6])):
            packet = copy.deepcopy(self.packet)
            mutate(packet)
            packet["packet_hash"] = fingerprint({key: value for key, value in packet.items() if key != "packet_hash"})
            with self.assertRaises(ValueError):
                serialize(packet, [dict(row, packet_hash=packet["packet_hash"]) for row in self.candidates])
        snapshot = serialize(self.packet, self.candidates)
        snapshot["slots"][0]["numeric"][0] = 99.
        with self.assertRaises(ValueError):
            tensorize([snapshot], self.encoder)

    def test_expired_before_draw_and_underflow_is_not_forced(self):
        selector = ContextualSelector(self.network, self.encoder, self.packet["owner"], "fixture")
        with self.assertRaises(TimeoutError):
            selector.choose(self.packet, self.candidates, time.monotonic() - 1.)
        self.assertEqual(selector.number, 0)
        selector.evaluate = lambda _: (torch.zeros(len(self.candidates)),
            torch.tensor([1.] + [0.] * (len(self.candidates) - 1)), torch.tensor(.2))
        choice = selector.choose(self.packet, self.candidates, time.monotonic() + 20.)
        self.assertEqual(choice["metadata"]["selection_kind"], "sampled")
        self.assertIsNotNone(choice["metadata"]["draw_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
