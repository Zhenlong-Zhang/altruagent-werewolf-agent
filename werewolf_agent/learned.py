"""Explicitly opted-in transfer of a simulator policy into one practice seat."""

import json
import math
from collections import Counter
from pathlib import Path
import re

from .beliefs import build_suspicion, known_roles, parse_role, seat_from_name
from .policy import WerewolfPolicy, _all_messages, _state_data, _target

FEATURES = {"suspicion", "seer_claim", "pressure", "activity"}
PHASES = {"vote", "night_wolf", "night_seer"}
SEER_CLAIM = re.compile(r"\b(?:i am|i'm|im)\s+(?:the\s+)?seer\b", re.I)


def read_policy(path, name):
    artifact = json.loads(Path(path).read_text())
    weights = artifact["policies"][name]
    if artifact["schema_version"] != 1 or set(weights) != PHASES:
        raise ValueError("Unexpected offline policy schema")
    if any(set(row) != FEATURES or not all(type(v) in (float, int) and math.isfinite(v)
            for v in row.values()) for row in weights.values()):
        raise ValueError("Invalid offline policy weights")
    return weights


class LearnedPolicy(WerewolfPolicy):
    def __init__(self, variant, weights):
        super().__init__()
        self._strategy_version = f"surrogate-{variant}"
        self.weights = weights

    def choose_action(self, state, context):
        if context.tournament_id is not None:
            return super().choose_action(state, context)
        game, players = _state_data(state)
        phase = game.get("phase")
        phase = "vote" if phase == "day_vote" else phase
        if phase not in self.weights:
            return super().choose_action(state, context)
        seat = context.seat_position
        role = parse_role(state.observation)
        known = known_roles(state.observation, seat)
        _all_messages(state, self._messages)
        scores = build_suspicion(state.observation, game, players, seat, self._messages)
        candidates = [a for a in state.legal_actions if _target(a) is not None]
        if phase == "vote" and role == "wolf":
            candidates = [a for a in candidates if known.get(_target(a)) != "wolf"]
        if phase == "vote" and role == "seer":
            confirmed = [a for a in candidates if known.get(_target(a)) == "wolf"]
            candidates = confirmed or candidates
        if phase == "night_seer":
            fresh = [a for a in candidates if _target(a) not in known]
            candidates = fresh or candidates
        if not candidates:
            return super().choose_action(state, context)
        public = [m for m in self._messages if m["type"] == "chat" and not m["recipients"]]
        activity = Counter(m["sender"] for m in public)
        seer_claims = {m["sender"] for m in public if SEER_CLAIM.search(m["content"])}
        ballots = Counter(
            seat_from_name(target, players)
            for round_info in game.get("vote_history", [])
            for target in round_info.get("votes", {}).values()
            if target != "abstain"
        )
        messages_high = max(activity[_target(a)] for a in candidates) or 1
        ballot_count = ballots.total() or 1
        weights = self.weights[phase]

        def rank(action):
            target = _target(action)
            features = {
                "suspicion": max(-1., min(1., scores.get(target, 0.))),
                "seer_claim": float(target in seer_claims),
                "pressure": ballots[target] / ballot_count,
                "activity": activity[target] / messages_high,
            }
            return sum(weights[k] * v for k, v in features.items()), -target

        chosen = max(candidates, key=rank)
        self._ensure_logger(context).decision(state, context, role, chosen, scores)
        return chosen
