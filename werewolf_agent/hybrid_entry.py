"""One configurable focal seat for official joint LLM and RL collection."""

import json
import os
from pathlib import Path

from .entry import MixedTable
from .hybrid import HybridPolicy
from .policy import WerewolfPolicy
from .variants import LLMPolicy

MODES = {"llm_select", "hybrid_collect", "hybrid_greedy"}


class HybridTable(MixedTable):
    def __init__(self, seat, mode, weights, assignments, opponents):
        super().__init__(seat, mode, weights, assignments)
        self.opponents = opponents

    def _policy(self, context):
        if self.policy is None:
            seat, mode = self.assignments.get(context.session_id, (self.seat, self.variant))
            if context.tournament_id is not None or seat is None:
                self.policy = WerewolfPolicy()
            elif context.seat_position == seat:
                self.policy = HybridPolicy(mode, self.weights)
                self.policy.opponent_pool = list(self.opponents)
            else:
                positions = [p for p in range(7) if p != seat]
                opponent = self.opponents[positions.index(context.seat_position)]
                if opponent == "rules":
                    self.policy = WerewolfPolicy()
                else:
                    self.policy = LLMPolicy(opponent)
                    self.policy._strategy_version = f"pool-{opponent}:{self.policy.model}"
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
    opponents = os.environ.get("WEREWOLF_HYBRID_OPPONENTS", "rules,rules,rules,rules,rules,rules").split(",")
    if len(opponents) != 6 or any(p not in {"rules", "direct", "belief"} for p in opponents):
        raise ValueError("Provide six opponent slots using rules, direct or belief")
    return HybridTable(seat, mode, os.environ.get("WEREWOLF_HYBRID_WEIGHTS"), assignments, opponents)
