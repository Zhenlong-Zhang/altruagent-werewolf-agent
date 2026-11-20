"""Hoffman2 Phase-B controller checks; no language model required."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from altruagent import LegalAction, TERMINATE_MESSAGING
from werewolf_agent.strategy_candidates_v2 import teacher_candidates, validate_delivery_candidate
from werewolf_agent.strategy_dialogue import DialogueController
from werewolf_agent.strategy_memory import StrategyMemory
from werewolf_agent.strategy_policy import StrategyPolicy


class ControllerChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.context = SimpleNamespace(session_id="owned-b-fixture", seat_position=6)
        self.owner = {"session_hash": hashlib.sha256(self.context.session_id.encode()).hexdigest()[:16], "seat": 6}
        self.path = Path(self.tmp.name) / "private.jsonl"
        self.memory = StrategyMemory(self.owner, self.path)
        self.addCleanup(self.memory.close)
        self.state = SimpleNamespace(observation="Your role: SEER\nYour investigations: Player2 is villager",
            phase="messaging", new_messages=[], terminated_messaging=[],
            raw={"messaging_seconds_left": 100., "players": [{"position": i, "name": f"Player{i}"} for i in range(7)],
                 "game_state": {"day": 2, "phase": "day_vote", "alive": [f"Player{i}" for i in range(7)], "dead": [], "vote_history": []}},
            legal_actions=[LegalAction(action_id="opaque:first", label="Vote Player1", input={}),
                           LegalAction(action_id="opaque:second", label="Vote Player2", input={}),
                           LegalAction(action_id="opaque:pass", label="Abstain", input={})])
        self.memory.observe_state(self.state, self.context, 10.)

    def packet(self):
        return self.memory.compile_packet(self.state, self.context, lambda _: 100, now=11.)

    def plans(self):
        return {"candidates": [{"intent": intent, "target": target, "recipient": None, "assertions": [], "text": ""}
            for intent, target in [("ask_evidence", 1), ("withhold", None), ("coordinate_vote", 4)]], "preferred_index": 2}

    def candidate(self, index=0):
        packet = self.packet()
        candidates, preferred = teacher_candidates(self.plans(), packet, "speech")
        self.assertEqual(preferred["index"], 2)
        candidate = DialogueController(self.memory).candidates(candidates, 11.)[index]
        validate_delivery_candidate(candidate, packet)
        return packet, candidate

    @staticmethod
    def message(seq, content, sender=6):
        return {"seq": seq, "sender": sender, "recipients": [], "type": "chat", "content": content, "day": 2}

    def test_ack_reject_unknown_and_restart_are_not_duplicate_delivery(self):
        packet, candidate = self.candidate()
        self.memory.prepare("one", 1, candidate, packet, 11.)
        self.memory.apply_delivery("one", {"error": "MESSAGE_DELIVERY_UNKNOWN"}, self.context, 12.)
        self.assertFalse(self.memory.can_chat(12.))
        self.memory.merge_visible([self.message(5, candidate["text"])], "authorized_refresh", True, 13.)
        self.memory.apply_delivery("one", {"accepted": True}, self.context, 14.)
        self.assertEqual(self.memory.current.accepted, 1)
        self.assertEqual(self.memory.assertions[0]["accepted_seq"], 5)
        self.memory.close()
        restored = StrategyMemory(self.owner, self.path)
        self.addCleanup(restored.close)
        self.assertEqual(restored.windows[2].accepted, 1)
        self.assertEqual(restored.assertions[0]["original_packet_hash"], packet["packet_hash"])
        self.assertFalse(restored.can_chat(14.))

    def test_rejections_consume_attempts_but_not_acks(self):
        for index in range(4):
            packet, candidate = self.candidate()
            self.memory.prepare(str(index), index, candidate, packet, 11.)
            self.memory.apply_delivery(str(index), {"accepted": False}, self.context, 12.)
        self.assertEqual(self.memory.current.accepted, 0)
        self.assertEqual(self.memory.current.attempted, 4)
        self.assertFalse(self.memory.can_chat(12.))
        self.assertEqual(self.memory.assertions, [])
        with self.assertRaisesRegex(ValueError, "Conflicting rejected/accepted"):
            self.memory.apply_delivery("0", {"accepted": True}, self.context, 13.)

    def test_three_acks_and_multiple_matches_do_not_manufacture_acceptance(self):
        for index in range(3):
            packet, candidate = self.candidate(index)
            self.memory.prepare(str(index), index, candidate, packet, 11.)
            self.memory.apply_delivery(str(index), {"accepted": True}, self.context, 12.)
        self.assertEqual(self.memory.current.accepted, 3)
        self.assertFalse(self.memory.can_chat(12.))
        self.memory.close()
        other = StrategyMemory(self.owner, Path(self.tmp.name) / "ambiguous.jsonl")
        self.addCleanup(other.close)
        other.observe_state(self.state, self.context, 10.)
        packet = other.compile_packet(self.state, self.context, lambda _: 100)
        candidates, _ = teacher_candidates(self.plans(), packet, "speech")
        other.prepare("unknown", 1, candidates[0], packet, 11.)
        other.merge_visible([self.message(7, candidates[0]["text"]), self.message(8, candidates[0]["text"])], "refresh", True, 12.)
        self.assertEqual(other.current.accepted, 0)
        self.assertIsNotNone(other.current.pending)
        self.assertFalse(other.can_chat(12.))

    def test_relevance_dedup_clock_and_owner(self):
        self.memory.current.consumed_seq = 2
        self.memory.merge_visible([self.message(1, "Player6 explain", 1), self.message(3, "Player6 explain", 1)], "refresh", True, 12.)
        self.assertFalse(self.memory.relevant_update())
        self.memory.merge_visible([self.message(4, "Player6 what is your vote?", 2)], "refresh", True, 12.)
        self.assertTrue(self.memory.relevant_update())
        self.memory.current.consumed_seq = 4
        self.memory.merge_visible([self.message(5, "Player 1 explain your evidence", 2)], "refresh", True, 12.)
        self.assertTrue(self.memory.relevant_update())
        self.assertFalse(self.memory.can_chat(109.))
        with self.assertRaises(ValueError):
            self.memory.observe_state(self.state, SimpleNamespace(session_id="other", seat_position=6), 12.)
        with self.assertRaises(ValueError):
            self.memory.merge_visible([dict(self.message(7, "private", 1), recipients=[2])], "refresh", True, 12.)

    def test_controller_opening_response_changed_commitment_then_stop(self):
        controller = DialogueController(self.memory)
        self.assertEqual(controller.should_generate(11.), (True, "opening"))
        packet, opening = self.candidate(2)
        self.memory.prepare("opening", 1, opening, packet, 11.)
        self.memory.apply_delivery("opening", {"accepted": True}, self.context, 12.)
        self.assertEqual(controller.should_generate(12.), (False, "no_new_relevant_evidence"))
        self.memory.merge_visible([self.message(1, "Player 1 please explain your vote", 1)], "refresh", True, 13.)
        self.assertEqual(controller.should_generate(13.), (True, "new_evidence"))
        packet, response = self.candidate(0)
        self.assertTrue(controller.admissible(response, 13.)[0])
        self.memory.prepare("response", 2, response, packet, 13.)
        self.memory.apply_delivery("response", {"accepted": True}, self.context, 14.)
        self.memory.merge_visible([self.message(2, "Player1 changed their stated vote", 2)], "refresh", True, 15.)
        self.assertTrue(controller.should_generate(15.)[0])
        packet = self.packet()
        candidates, _ = teacher_candidates(self.plans(), packet, "speech")
        self.assertFalse(controller.admissible(candidates[2], 15.)[0])
        changed = self.plans()
        changed["candidates"][2]["target"] = 1
        candidates, _ = teacher_candidates(changed, packet, "speech")
        self.assertTrue(controller.admissible(candidates[2], 15.)[0])
        self.memory.prepare("changed", 3, candidates[2], packet, 15.)
        self.memory.apply_delivery("changed", {"accepted": True}, self.context, 16.)
        self.assertFalse(controller.should_generate(16.)[0])
        self.assertEqual([row["target"] for row in self.memory.semantic_assertions()], [4, 1])

    def test_private_buddy_ack_uses_shared_quota_and_blocks_second_private(self):
        self.state.observation = "Your role: WOLF\nYour fellow wolves: Player2"
        packet = self.packet()
        plans = self.plans()
        plans["candidates"][0] = {"intent": "coordinate_private", "target": 1, "recipient": 2, "assertions": [], "text": ""}
        candidates, _ = teacher_candidates(plans, packet, "speech")
        controller = DialogueController(self.memory)
        self.assertTrue(controller.admissible(candidates[0], 11.)[0])
        self.memory.prepare("private", 1, candidates[0], packet, 11.)
        self.memory.apply_delivery("private", {"accepted": True, "remaining": {"this_turn": 4}}, self.context, 12.)
        self.memory.merge_visible([dict(self.message(1, "Player1 agreed", 2), recipients=[6])], "refresh", True, 13.)
        self.assertTrue(controller.should_generate(13.)[0])
        self.assertEqual(controller.admissible(candidates[0], 13.), (False, "private_coordination_cap"))
        self.assertTrue(controller.admissible(candidates[1], 13.)[0])
        self.assertEqual(self.memory.current.accepted, 1)
        self.assertEqual(self.memory.current.remaining, 4)

    def test_typed_pending_and_phase_flip(self):
        packet, candidate = self.candidate()
        self.memory.prepare("one", 1, candidate, packet, 11.)
        invalid = copy.deepcopy(self.memory.current.pending)
        invalid["status"] = {"winner": "wolves"}
        with self.assertRaises(ValueError):
            self.memory._validate_pending(invalid)
        self.state.phase = "moving"
        self.memory.observe_state(self.state, self.context, 12.)
        self.assertFalse(self.memory.can_chat(12.))
        self.assertEqual(self.memory.current.pending["owner"]["day"], 2)

    def test_long_memory_keeps_fact_and_own_commitment_without_fulltext(self):
        packet, candidate = self.candidate(2)
        self.memory.prepare("one", 1, candidate, packet, 11.)
        self.memory.apply_delivery("one", {"accepted": True}, self.context, 12.)
        self.memory.merge_visible([self.message(i, "Player2 is wolf. " + "rumor " * 100, i % 6) for i in range(100)], "refresh", True, 13.)
        packet = self.packet()
        self.assertEqual(next(f["alignment"] for f in packet["verified_facts"] if f["id"] == "check:2"), "nonwolf")
        self.assertEqual(packet["self_assertions"][0]["target"], 4)
        self.assertLessEqual(len(packet["attributed_claims"]), 28)
        self.assertTrue(packet["provenance"]["dropped_claim_count"] > 0)
        self.assertTrue(all(len(row["text"]) <= 256 for row in packet["attributed_claims"]))

    def test_teacher_preference_fulllegal_and_invalid_indices(self):
        packet = self.packet()
        actions, preference = teacher_candidates({"action_id": "opaque:pass"}, packet, "action")
        self.assertEqual(len(actions), 3)
        self.assertEqual(preference["index"], 2)
        for bad in (True, -1, 3, None):
            with self.assertRaises(ValueError):
                teacher_candidates(dict(self.plans(), preferred_index=bad), packet, "speech")

    def test_duplicate_ack_semantics_keep_original_and_latest_refs(self):
        packet, candidate = self.candidate(2)
        self.memory.prepare("one", 1, candidate, packet, 11.)
        self.memory.apply_delivery("one", {"accepted": True}, self.context, 12.)
        original = copy.deepcopy(self.memory.assertions[0])
        for index in range(70):
            self.memory.assertions.append(dict(original, delivery_id=f"duplicate:{index}", accepted_seq=100 + index))
        retained = self.memory.semantic_assertions()
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0]["delivery_id"], "one")
        self.assertEqual(retained[-1]["accepted_seq"], 169)
        self.assertEqual(self.packet()["self_assertions"], retained)
        self.memory.assertions.insert(1, dict(original, delivery_id="intermediate-target", target=3, accepted_seq=90))
        self.assertEqual([row["target"] for row in self.memory.semantic_assertions()], [4, 3, 4])

    def test_policy_preserves_valid_sample_but_gates_late_request(self):
        self.memory.close()
        clock = [10.]
        answer = self.plans()
        backend = SimpleNamespace(complete=lambda *args: {"text": "<answer>" + json.dumps(answer) + "</answer>", "provenance": {}})
        seen = {}
        def choose(packet, candidates, **kwargs):
            seen.update(kwargs)
            clock[0] = 45.
            return {"index": 2, "probabilities": [.1, .1, .6, .1, .1], "log_probability": __import__('math').log(.6),
                    "value": .25, "checkpoint_id": "fixture", "metadata": {}}
        policy = StrategyPolicy(self.context, backend, lambda _: 100, self.path, selector=SimpleNamespace(choose=choose), now=lambda: clock[0])
        self.addCleanup(policy.memory.close)
        self.assertIs(policy.choose_message(self.state, self.context), TERMINATE_MESSAGING)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        request = next(row for row in rows if row["event"] == "strategy_request")
        self.assertEqual(request["requested_index"], 2)
        self.assertEqual(request["behavior_selection"]["value"], .25)
        self.assertTrue(request["executor_overridden"])
        self.assertEqual(seen["decision_id"], 1)

    def test_invalid_selector_payload_is_not_recorded_as_categorical(self):
        self.memory.close()
        answer = self.plans()
        backend = SimpleNamespace(complete=lambda *args: {"text": "<answer>" + json.dumps(answer) + "</answer>", "provenance": {}})
        selector = SimpleNamespace(choose=lambda *args, **kwargs: {"index": 0, "probabilities": [float("nan")] * 5,
            "value": 0., "log_probability": 0., "metadata": {}, "checkpoint_id": "bad"})
        policy = StrategyPolicy(self.context, backend, lambda _: 100, self.path, selector=selector, now=lambda: 10.)
        self.addCleanup(policy.memory.close)
        self.assertIs(policy.choose_message(self.state, self.context), TERMINATE_MESSAGING)
        request = next(json.loads(line) for line in self.path.read_text().splitlines() if json.loads(line)["event"] == "strategy_request")
        self.assertIsNone(request["behavior_selection"])
        self.assertEqual(request["behavior_kind"], "forced_fallback")
        self.assertEqual(request["teacher_choice"]["index"], 2)


if __name__ == "__main__":
    unittest.main()
