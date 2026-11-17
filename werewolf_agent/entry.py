"""Starter entry point with an optional experimental policy on one seat."""

import os

from .policy import WerewolfPolicy
from .variants import LLMPolicy


class MixedTable:
    def __init__(self, seat, variant):
        self.seat = seat
        self.variant = variant
        self.policy = None

    def _policy(self, context):
        if self.policy is None:
            if context.seat_position == self.seat and self.variant != "rules":
                self.policy = LLMPolicy(self.variant)
                self.policy._strategy_version = f"{self.variant}:{self.policy.model}"
            else:
                self.policy = WerewolfPolicy()
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
    if variant not in {"direct", "belief"}:
        raise ValueError("WEREWOLF_CANDIDATE must be rules, direct or belief")
    seat = int(os.environ.get("WEREWOLF_CANDIDATE_SEAT", "0"))
    if seat not in range(7):
        raise ValueError("WEREWOLF_CANDIDATE_SEAT must be between 0 and 6")
    if os.environ.get("WEREWOLF_LOCAL_MODEL") != "1" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("The selected LLM candidate requires OPENAI_API_KEY")
    return MixedTable(seat, variant)
