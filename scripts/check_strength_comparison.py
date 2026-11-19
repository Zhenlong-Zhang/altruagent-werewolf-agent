"""Run comparison routing and result-accounting regressions on Hoffman2."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.report_hybrid import read_rows, report
from werewolf_agent.hybrid import HybridPolicy
from werewolf_agent.hybrid_entry import create_agent
from werewolf_agent.collection_entry import create_agent as create_collector
from werewolf_agent.selector_compare_entry import create_agent as create_checkpoint_comparison
from werewolf_agent.policy_compare_entry import create_agent as create_policy_comparison
from scripts.summarize_strength import expected_assignment


class ComparisonChecks(unittest.TestCase):
    def test_complete_policy_profiles_route_mode_guard_and_checkpoint_before_observation(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            model=root/'FakeModel'
            model.mkdir()
            (model/'config.json').write_text('{}')
            checkpoint=Path(__file__).resolve().parents[1]/'outputs/hybrid/qwen3-8b-round1/selector.json'
            changed=json.loads(checkpoint.read_text())
            changed['weights'][0]+=.05
            adapted=root/'profile-adapted.json'
            adapted.write_text(json.dumps(changed))
            profiles=[dict(name='incumbent',mode='hybrid_greedy',guard='none',weights=str(checkpoint)),
                      dict(name='guarded',mode='hybrid_greedy',guard='verified-innocent-v1',weights=str(checkpoint)),
                      dict(name='sampled',mode='hybrid_collect',guard='verified-innocent-v1',weights=str(checkpoint)),
                      dict(name='adapted',mode='hybrid_collect',guard='verified-innocent-v1',weights=str(adapted))]
            seed='profile-test'
            names=[profile['name'] for profile in profiles]
            sessions={}
            for index in range(128):
                session=f'profile-{index}'
                seat,arm=expected_assignment(session,names,seed,'wolf-policy-v1')
                sessions.setdefault(arm,(session,seat))
                if len(sessions)==4:
                    break
            self.assertEqual(len(sessions),4)
            for arm,(session,expected_seat) in sessions.items():
                with patch.dict(os.environ,{'WEREWOLF_MODEL_PATH':str(model),
                    'WEREWOLF_LOG_DIR':str(root/'logs'),'WEREWOLF_POLICY_PROFILES':json.dumps(profiles),
                    'WEREWOLF_POLICY_COMPARE_SEED':seed,'WEREWOLF_PROPOSER':'ids'},clear=True):
                    focal=[]
                    for seat in range(7):
                        context=SimpleNamespace(session_id=session,tournament_id=None,game_type='werewolf',seat_position=seat)
                        policy=create_policy_comparison()._policy(context)
                        if isinstance(policy,HybridPolicy):
                            focal.append(seat)
                            self.addCleanup(policy._model.client.close)
                            row=json.loads(policy._logger.path.read_text().splitlines()[0])
                            expected=next(p for p in profiles if p['name']==arm)
                            self.assertEqual(row['arm'],arm)
                            self.assertEqual(row['assignment_version'],'wolf-policy-v1')
                            self.assertEqual(policy.mode,expected['mode'])
                            self.assertEqual(policy._model.evidence_guard,expected['guard'])
                            self.assertEqual(policy._model.proposer,'ids')
                            from werewolf_agent.selector import Selector
                            expected_id=Selector(expected['mode'],expected['weights']).checkpoint_id
                            self.assertEqual(policy.selector.checkpoint_id,expected_id)
                            self.assertEqual(row['checkpoint_id'],expected_id)
                            self.assertTrue(policy.allow_tournament)
                            self.assertNotIn('role_for_eval_only',row)
                    self.assertEqual(focal,[expected_seat])
            self.assertEqual(len(report(root/'logs')),4)

    def test_complete_policy_comparison_rejects_ambiguous_profiles(self):
        for profiles in ([],[dict(name='only')],
                         [dict(name='a',mode='hybrid_greedy',guard='none',weights=None),
                          dict(name='b',mode='hybrid_greedy',guard='none',weights=None)],
                         [dict(name='same',mode='hybrid_greedy',guard='none',weights='missing')]*2):
            with patch.dict(os.environ,{'WEREWOLF_POLICY_PROFILES':json.dumps(profiles)},clear=True):
                with self.assertRaises((ValueError,FileNotFoundError)):
                    create_policy_comparison()

    def test_collection_randomizes_one_focal_seat_before_observations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "FakeModel"
            model.mkdir()
            (model / "config.json").write_text("{}")
            checkpoint = Path(__file__).resolve().parents[1] / "outputs/hybrid/qwen3-8b-round1/selector.json"
            with patch.dict(os.environ, {"WEREWOLF_MODEL_PATH": str(model),
                "WEREWOLF_LOG_DIR": str(root / "logs"), "WEREWOLF_HYBRID_WEIGHTS": str(checkpoint),
                "WEREWOLF_HYBRID_MODE": "hybrid_collect", "WEREWOLF_PROPOSER": "grounded"}, clear=True):
                focal = []
                for seat in range(7):
                    context = SimpleNamespace(session_id="collection-session", tournament_id=None,
                                              game_type="werewolf", seat_position=seat)
                    policy = create_collector()._policy(context)
                    if isinstance(policy, HybridPolicy):
                        focal.append(seat)
                        self.addCleanup(policy._model.client.close)
                        self.assertEqual(policy.mode, "hybrid_collect")
                self.assertEqual(len(focal), 1)
                rows = [json.loads(line) for path in (root / "logs").glob("*.jsonl") for line in path.read_text().splitlines()]
                self.assertEqual(rows[0]["event"], "collection_assignment")
                self.assertEqual(rows[0]["assignment_version"], "wolf-collect-v1")
                self.assertNotIn("role_for_eval_only", rows[0])
                self.assertEqual(len(report(root / "logs")), 1)

    def test_checkpoint_comparison_routes_exact_checkpoint_with_fixed_proposer(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "FakeModel"
            model.mkdir()
            (model / "config.json").write_text("{}")
            checkpoint = Path(__file__).resolve().parents[1] / "outputs/hybrid/qwen3-8b-round1/selector.json"
            changed = json.loads(checkpoint.read_text())
            changed["weights"][0] += .01
            adapted = root / "adapted.json"
            adapted.write_text(json.dumps(changed))
            with patch.dict(os.environ, {"WEREWOLF_MODEL_PATH": str(model),
                "WEREWOLF_LOG_DIR": str(root / "logs"), "WEREWOLF_HYBRID_MODE": "hybrid_collect",
                "WEREWOLF_PROPOSER": "grounded", "WEREWOLF_SELECTOR_CHECKPOINTS": json.dumps({
                    "incumbent": str(checkpoint), "adapted": str(adapted)})}, clear=True):
                for session in ("head-session-a", "head-session-b"):
                    focal = []
                    for seat in range(7):
                        context = SimpleNamespace(session_id=session, tournament_id=None,
                                                  game_type="werewolf", seat_position=seat)
                        policy = create_checkpoint_comparison()._policy(context)
                        if isinstance(policy, HybridPolicy):
                            focal.append(policy)
                            self.addCleanup(policy._model.client.close)
                            row = json.loads(policy._logger.path.read_text().splitlines()[0])
                            from werewolf_agent.selector import Selector
                            expected = checkpoint if row["arm"] == "incumbent" else adapted
                            self.assertEqual(row["checkpoint_id"], Selector("hybrid_collect", expected).checkpoint_id)
                            self.assertEqual(row["assignment_version"], "wolf-selector-v1")
                            self.assertEqual(policy._model.proposer, "grounded")
                    self.assertEqual(len(focal), 1)

    def test_collection_rejects_wrong_mode_before_connecting(self):
        with patch.dict(os.environ, {"WEREWOLF_HYBRID_MODE": "hybrid_greedy"}, clear=True):
            with self.assertRaisesRegex(ValueError, "hybrid_collect"):
                create_collector()

    def test_canary_uses_exact_competition_factory_for_one_focal_seat(self):
        with TemporaryDirectory() as directory:
            backend = Path(directory) / "FakeModel"
            backend.mkdir()
            (backend / "config.json").write_text("{}")
            with patch.dict(os.environ, {
                "WEREWOLF_HYBRID_COMPARE_PROPOSERS": "ids,grounded",
                "WEREWOLF_HYBRID_MODE": "llm_select", "WEREWOLF_MODEL_PATH": str(backend),
                "WEREWOLF_LOG_DIR": str(Path(directory) / "logs"),
                "WEREWOLF_FOCAL_COMPETITION_FACTORY": "1",
            }, clear=True):
                focal = []
                for seat in range(7):
                    context = SimpleNamespace(session_id="competition-canary", tournament_id=None,
                                              game_type="werewolf", seat_position=seat)
                    policy = create_agent()._policy(context)
                    if isinstance(policy, HybridPolicy):
                        focal.append(policy)
                        self.addCleanup(policy._model.client.close)
                        self.assertTrue(policy.allow_tournament)
                        self.assertTrue(policy._strategy_version.startswith("competition:"))
                self.assertEqual(len(focal), 1)

    def test_seat_and_arm_are_shared_before_any_observation(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {
            "WEREWOLF_HYBRID_COMPARE_PROPOSERS": "ids,reasoned",
            "WEREWOLF_HYBRID_MODE": "hybrid_greedy", "WEREWOLF_MODEL_PATH": "/unused",
            "WEREWOLF_LOG_DIR": directory,
        }, clear=True):
            focal = []
            for seat in range(7):
                context = SimpleNamespace(session_id="frozen-comparison-session", tournament_id=None,
                                          game_type="werewolf", seat_position=seat)
                policy = create_agent()._policy(context)
                if isinstance(policy, HybridPolicy):
                    focal.append(seat)
            self.assertEqual(len(focal), 1)
            paths = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            assignment = json.loads(paths[0].read_text())
            self.assertEqual(assignment["seat"], focal[0])
            self.assertIn(assignment["arm"], {"ids", "reasoned"})
            self.assertEqual(assignment["event"], "comparison_assignment")
            self.assertNotIn("role_for_eval_only", assignment)

    def test_default_practice_tournament_guard(self):
        with patch.dict(os.environ, {"WEREWOLF_MODEL_PATH": "/unused"}, clear=True):
            context = SimpleNamespace(session_id="tournament", tournament_id="t1",
                                      game_type="werewolf", seat_position=0)
            self.assertNotIsInstance(create_agent()._policy(context), HybridPolicy)

    def test_invalid_comparison_fails(self):
        with patch.dict(os.environ, {"WEREWOLF_MODEL_PATH": "/unused",
                                    "WEREWOLF_HYBRID_COMPARE_PROPOSERS": "ids,ids"}, clear=True):
            with self.assertRaises(ValueError):
                create_agent()

    def test_no_choice_game_is_retained(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "seat.jsonl"
            rows = [dict(event="comparison_assignment", match_hash="m", seat=3,
                         mode="hybrid_greedy", arm="reasoned", checkpoint_id="frozen"),
                    dict(event="game_result", score=-1, role_for_eval_only="villager",
                         termination_reason="completed")]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            games = report(Path(directory))
            self.assertEqual(len(games), 1)
            self.assertEqual(games[0]["decisions"], 0)
            self.assertEqual(games[0]["score"], -1)
            self.assertFalse(games[0]["incomplete_tail"])

    def test_partial_live_tail_is_visible(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            path.write_text('{"event":"ok"}\n{"event":')
            rows, incomplete = read_rows(path)
            self.assertTrue(incomplete)
            self.assertEqual(rows, [{"event": "ok"}])
            path.write_text('{"broken":}\n{"event":"ok"}\n')
            with self.assertRaises(json.JSONDecodeError):
                read_rows(path)


if __name__ == "__main__":
    unittest.main()
