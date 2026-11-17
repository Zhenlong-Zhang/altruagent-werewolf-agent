"""One configurable focal seat for official joint LLM and RL collection."""

import json
import os
from pathlib import Path

from .entry import MixedTable
from .hybrid import HybridPolicy
from .policy import WerewolfPolicy

MODES = {"llm_select", "hybrid_collect", "hybrid_greedy"}


class HybridTable(MixedTable):
    def _policy(self, context):
        if self.policy is None:
            seat, mode = self.assignments.get(context.session_id, (self.seat, self.variant))
            if context.tournament_id is None and context.seat_position == seat:
                self.policy = HybridPolicy(mode, self.weights)
            else:
                self.policy = WerewolfPolicy()
        return self.policy

    def on_message_result(self, result, context):
        callback = getattr(self._policy(context), "on_message_result", None)
        if callable(callback):
            callback(result, context)


def create_agent():
    mode = os.environ.get("WEREWOLF_HYBRID_MODE", "llm_select")
    seat = int(os.environ.get("WEREWOLF_CANDIDATE_SEAT", "0"))
    if mode not in MODES or seat not in range(7):
        raise ValueError("Invalid hybrid mode or seat")
    if not os.environ.get("WEREWOLF_CPU_MODEL"):
        raise ValueError("Set WEREWOLF_CPU_MODEL inside the Hoffman2 job")
    path = os.environ.get("WEREWOLF_HYBRID_MANIFEST")
    assignments = {}
    if path:
        assignments = json.loads(Path(path).read_text())
        if any(s not in range(7) or m not in MODES for s, m in assignments.values()):
            raise ValueError("Invalid hybrid assignment manifest")
        seat = None
    return HybridTable(seat, mode, os.environ.get("WEREWOLF_HYBRID_WEIGHTS"), assignments)
