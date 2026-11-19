"""Result-denominator checks, runnable without model weights under Slurm."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.summarize_strength import expected_assignment, group_summary, main, wilson


class AccountingChecks(unittest.TestCase):
    def test_matching_seat_and_arm_do_not_hide_wrong_assignment_namespace(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root/'logs'
            logs.mkdir()
            session='same-seat-wrong-namespace'
            seat,arm=expected_assignment(session,['ids','reasoned'],'r03-frozen-v1')
            row=dict(event='comparison_assignment',match_hash=hashlib.sha256(session.encode()).hexdigest()[:16],
                     seat=seat,arm=arm,mode='hybrid_greedy',assignment_version='wrong-version',assignment_seed='wrong-seed')
            (logs/'game.jsonl').write_text(json.dumps(row)+'\n')
            census,out=root/'census.json',root/'summary.json'
            census.write_text(json.dumps([dict(match_id=session)]))
            with patch.object(sys,'argv',['summary',str(logs),'--census',str(census),'--out',str(out)]):
                main()
            errors={problem['error'] for problem in json.loads(out.read_text())['protocol_problems']}
            self.assertEqual(errors,{'assignment_version_mismatch','assignment_seed_mismatch'})

    def test_unresolved_and_stale_are_never_losses_or_wins(self):
        group = group_summary([
            dict(terminal_count=1, score=1, termination_reason="completed", role="seer"),
            dict(terminal_count=1, score=0, termination_reason="testing_stale"),
            dict(terminal_count=2, score=1, termination_reason="completed"),
            dict(terminal_count=1, score=1, termination_reason="unrecognized_server_reason"),
            {},
        ])
        self.assertEqual((group['assigned'], group['resolved'], group['wins']), (5, 1, 1))
        self.assertEqual(group['all_assigned_win_bounds'], [.2, 1.])
        self.assertIsNone(group['role_standardized_win_rate'])

    def test_role_standardization_and_wilson(self):
        rows = [dict(terminal_count=1, score=score, termination_reason="completed", role=role)
                for role, score in [('villager', 1), ('seer', -1), ('wolf', -1)]]
        self.assertAlmostEqual(group_summary(rows)['role_standardized_win_rate'], 4/7)
        low, high = wilson(1, 3)
        self.assertLess(low, 1/3)
        self.assertGreater(high, 1/3)
        self.assertIsNone(wilson(0, 0))

    def test_missing_worker_is_retained_and_duplicate_receipt_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root/'logs'
            logs.mkdir()
            census, out = root/'census.json', root/'summary.json'
            census.write_text(json.dumps([dict(match_id='unconnected')]))
            args = ['summarize_strength', str(logs), '--census', str(census), '--out', str(out)]
            with patch.object(sys, 'argv', args):
                main()
            result = json.loads(out.read_text())
            self.assertEqual(len(result['games']), 1)
            self.assertEqual(sum(arm['unresolved'] for arm in result['arms'].values()), 1)
            census.write_text(json.dumps([dict(match_id='duplicate')]*2))
            with patch.object(sys, 'argv', args), self.assertRaisesRegex(ValueError, 'Duplicate'):
                main()

    def test_receipt_assignment_mismatch_is_flagged(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root/'logs'
            logs.mkdir()
            session = 'assigned-session'
            seat, arm = expected_assignment(session, ['ids', 'reasoned'], 'r03-frozen-v1')
            row = dict(event='comparison_assignment', match_hash=hashlib.sha256(session.encode()).hexdigest()[:16],
                       seat=(seat+1)%7, arm=arm, mode='hybrid_greedy')
            (logs/'game.jsonl').write_text(json.dumps(row)+'\n')
            census, out = root/'census.json', root/'summary.json'
            census.write_text(json.dumps([dict(match_id=session)]))
            with patch.object(sys, 'argv', ['summary', str(logs), '--census', str(census), '--out', str(out)]):
                main()
            self.assertEqual(json.loads(out.read_text())['protocol_problems'][0]['error'], 'assignment_mismatch')


if __name__ == '__main__':
    unittest.main()
