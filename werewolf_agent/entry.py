"""Starter entry point with an optional experimental policy on one seat."""

import hashlib
import os

from .learned import LearnedPolicy, read_policy
from .policy import WerewolfPolicy
from .variants import LLMPolicy

COMPARISON_POLICIES = ("rules", "direct", "belief", "search", "reinforce")


class MixedTable:
    def __init__(self, seat, variant, weights=None):
        self.seat = seat
        self.variant = variant
        self.policy = None
        self.weights = weights

    def _policy(self, context):
        if self.policy is None:
            seat, variant = self.seat, self.variant
            if variant == "compare":
                seed = hashlib.sha256(context.session_id.encode()).digest()
                seat = int.from_bytes(seed[:8], "big") % 7
                variant = COMPARISON_POLICIES[int.from_bytes(seed[8:16], "big") % len(COMPARISON_POLICIES)]
            if context.seat_position != seat or context.tournament_id is not None:
                self.policy = WerewolfPolicy()
            elif variant == "rules":
                self.policy = WerewolfPolicy()
                self.policy._strategy_version = "focal-rules"
            elif variant in {"search", "reinforce"}:
                weights = self.weights if self.variant != "compare" else self.weights[variant]
                self.policy = LearnedPolicy(variant, weights)
            else:
                self.policy = LLMPolicy(variant)
                self.policy._strategy_version = f"{variant}:{self.policy.model}"
        return self.policy

    def choose_action(self, state, context):
        return self._policy(context).choose_action(state, context)

    def choose_message(self, state, context):
        return self._policy(context).choose_message(state, context)

    def on_action_result(self, result, context):
        self._policy(context).on_action_result(result, context)

    def on_game_result(self, state, context):
        self._policy(context).on_game_result(state, context)


def create_agent():
    variant = os.environ.get("WEREWOLF_CANDIDATE", "rules")
    if variant == "rules":
        return WerewolfPolicy()
    if variant not in {*COMPARISON_POLICIES, "compare"}:
        raise ValueError("WEREWOLF_CANDIDATE must be rules, direct, belief, search, reinforce or compare")
    seat = int(os.environ.get("WEREWOLF_CANDIDATE_SEAT", "0"))
    if seat not in range(7):
        raise ValueError("WEREWOLF_CANDIDATE_SEAT must be between 0 and 6")
    if variant in {"search", "reinforce"}:
        path = os.environ.get("WEREWOLF_WEIGHTS", "outputs/offline/weights.json")
        return MixedTable(seat, variant, read_policy(path, variant))
    if not (os.environ.get("WEREWOLF_CPU_MODEL") or os.environ.get("WEREWOLF_LOCAL_MODEL") == "1"
            or os.environ.get("OPENAI_API_KEY")):
        raise ValueError("The selected candidate requires a model backend")
    if variant == "compare":
        path = os.environ.get("WEREWOLF_WEIGHTS", "outputs/offline/weights.json")
        return MixedTable(seat, variant, {name: read_policy(path, name) for name in ("search", "reinforce")})
    return MixedTable(seat, variant)
