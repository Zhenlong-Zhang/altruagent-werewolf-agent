"""One configurable focal seat for official joint LLM and RL collection."""

import json
import hashlib
import os
from pathlib import Path

from .entry import MixedTable
from .hybrid import HybridPolicy
from .policy import WerewolfPolicy
from .variants import LLMPolicy

MODES = {"llm_select", "hybrid_collect", "hybrid_greedy"}


class HybridTable(MixedTable):
    def __init__(self, seat, mode, weights, assignments, opponents, proposers=(), seed="strength-v1", competition_factory=False,
                 assignment_version="wolf-compare-v1", assignment_event="comparison_assignment", checkpoint_choices=()):
        super().__init__(seat, mode, weights, assignments)
        self.opponents = opponents
        self.proposers = proposers
        self.comparison_seed = seed
        self.competition_factory = competition_factory
        self.assignment_version = assignment_version
        self.assignment_event = assignment_event
        self.checkpoint_choices = checkpoint_choices

    def _policy(self, context):
        if self.policy is None:
            seat, mode = self.assignments.get(context.session_id, (self.seat, self.variant))
            proposer = None
            weights = self.weights
            arm = None
            if self.proposers or self.checkpoint_choices:
                digest = hashlib.sha256(
                    f"{self.assignment_version}:{self.comparison_seed}:{context.session_id}".encode()
                ).digest()
                seat = int.from_bytes(digest[:8], "big") % 7
                if self.checkpoint_choices:
                    arm, weights = self.checkpoint_choices[int.from_bytes(digest[8:16], "big") % len(self.checkpoint_choices)]
                    proposer = os.environ.get("WEREWOLF_PROPOSER", "grounded")
                else:
                    proposer = self.proposers[int.from_bytes(digest[8:16], "big") % len(self.proposers)]
                    arm = proposer
            if context.tournament_id is not None or seat is None:
                self.policy = WerewolfPolicy()
            elif context.seat_position == seat:
                if proposer is not None:
                    os.environ["WEREWOLF_PROPOSER"] = proposer
                if self.competition_factory:
                    from .competition_entry import create_agent as create_competitor
                    if weights is not None:
                        os.environ["WEREWOLF_HYBRID_WEIGHTS"] = str(weights)
                    self.policy = create_competitor()
                else:
                    self.policy = HybridPolicy(mode, weights)
                self.policy.opponent_pool = list(self.opponents)
                if proposer is not None:
                    self.policy._ensure_logger(context)._write({
                        "event": self.assignment_event, "seat": seat, "mode": mode,
                        "arm": arm, "assignment_version": self.assignment_version,
                        "assignment_seed": self.comparison_seed,
                        "checkpoint_id": self.policy.selector.checkpoint_id
                            if self.policy.selector else "llm_direct",
                        "opponent_pool": list(self.opponents),
                        "competition_factory": self.competition_factory,
                    })
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
    if not (os.environ.get("WEREWOLF_MODEL_PATH") or os.environ.get("WEREWOLF_CPU_MODEL")):
        raise ValueError("Set WEREWOLF_MODEL_PATH inside the Hoffman2 job")
    validator = LLMPolicy("hybrid")
    validator.client.close()
    path = os.environ.get("WEREWOLF_HYBRID_MANIFEST")
    proposers = tuple(p.strip() for p in os.environ.get(
        "WEREWOLF_HYBRID_COMPARE_PROPOSERS", "").split(",") if p.strip())
    if proposers and (len(proposers) < 2 or len(set(proposers)) != len(proposers)
                      or any(p not in {"ids", "judge", "reasoned", "grounded"} for p in proposers) or path):
        raise ValueError("Comparison requires distinct valid proposers and no assignment manifest")
    if proposers and mode == "hybrid_collect":
        raise ValueError("Run frozen comparison separately from on-policy collection")
    assignments = {}
    if path:
        assignments = json.loads(Path(path).read_text())
        if any(s not in range(7) or m not in MODES for s, m in assignments.values()):
            raise ValueError("Invalid hybrid assignment manifest")
        seat = None
    opponents = os.environ.get("WEREWOLF_HYBRID_OPPONENTS", "rules,rules,rules,rules,rules,rules").split(",")
    if len(opponents) != 6 or any(p not in {"rules", "direct", "belief"} for p in opponents):
        raise ValueError("Provide six opponent slots using rules, direct or belief")
    return HybridTable(seat, mode, os.environ.get("WEREWOLF_HYBRID_WEIGHTS"), assignments, opponents,
                       proposers, os.environ.get("WEREWOLF_HYBRID_COMPARE_SEED", "strength-v1"),
                       os.environ.get("WEREWOLF_FOCAL_COMPETITION_FACTORY") == "1")
