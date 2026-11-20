"""Dedicated opt-in strategy entry. No legacy policy or global SDK mutation."""

import hashlib
import os
from pathlib import Path

from .strategy_backend import GenerationClient, LocalTokenCounter
from .strategy_policy import StrategyPolicy


class StrategyTable:
    requires_visible_transcript_flush = True

    def __init__(self, backend, counter, journal_dir, selector_factory=None, *, policy_options=None, runtime_metadata=None):
        self.backend, self.counter = backend, counter
        self.directory = Path(journal_dir)
        self.selector_factory = selector_factory
        self.policy_options = dict(policy_options or {})
        self.runtime_metadata = runtime_metadata
        self.policies = {}

    def strategy_transport_policy(self, context):
        if type(context.seat_position) is not int or context.seat_position not in range(7):
            raise ValueError("Strategy entry requires the assigned acting seat")
        key = (hashlib.sha256(context.session_id.encode()).hexdigest()[:16], context.seat_position)
        if key not in self.policies:
            selector = self.selector_factory(context) if self.selector_factory else None
            self.policies[key] = StrategyPolicy(context, self.backend, self.counter,
                self.directory / f"{key[0]}_seat{key[1]}.jsonl", selector=selector, **self.policy_options)
            if self.runtime_metadata is not None:
                policy = self.policies[key]
                import json
                previous = [json.loads(line) for line in policy.memory.path.read_text().splitlines()
                            if json.loads(line).get("event") == "strategy_runtime"] if policy.memory.path.exists() else []
                if previous:
                    expected = {"event": "strategy_runtime", "owner": policy.memory.owner, **self.runtime_metadata}
                    if len(previous) != 1 or previous[0] != expected:
                        policy.memory.close()
                        del self.policies[key]
                        raise ValueError("Reopened owner journal has a different runtime identity")
                elif policy.number:
                    policy.memory.close()
                    del self.policies[key]
                    raise ValueError("Existing requests lack their predeclared runtime identity")
                else:
                    policy.memory.record("strategy_runtime", **self.runtime_metadata)
        return self.policies[key]

    def choose_action(self, state, context):
        return self.strategy_transport_policy(context).choose_action(state, context)

    def choose_message(self, state, context):
        return self.strategy_transport_policy(context).choose_message(state, context)

    def on_action_result(self, result, context):
        self.strategy_transport_policy(context).on_action_result(result, context)

    def on_game_result(self, state, context):
        self.strategy_transport_policy(context).on_game_result(state, context)


def create_agent():
    return StrategyTable(GenerationClient(os.environ["WEREWOLF_STRATEGY_SOCKET"]),
        LocalTokenCounter(os.environ["WEREWOLF_STRATEGY_TOKENIZER"], os.environ["WEREWOLF_STRATEGY_REVISION"]),
        os.environ["WEREWOLF_STRATEGY_JOURNAL_DIR"])
