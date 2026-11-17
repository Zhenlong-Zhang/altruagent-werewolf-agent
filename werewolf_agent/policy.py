"""Interpretable role-aware baseline policy for AltruAgent Werewolf."""

from __future__ import annotations

import re
from typing import Any

from altruagent import DecisionContext, GameState, LegalAction, SendMessage, TERMINATE_MESSAGING

from .beliefs import PLAYER_RE, build_suspicion, known_roles, parse_role
from .logging import DecisionLogger


def _state_data(state: GameState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = state.raw if isinstance(state.raw, dict) else {}
    game_state = raw.get("game_state") if isinstance(raw.get("game_state"), dict) else {}
    players = raw.get("players") if isinstance(raw.get("players"), list) else []
    return game_state, players


def _target(action: LegalAction) -> int | None:
    if action.label:
        match = PLAYER_RE.search(action.label)
        if match:
            return int(match.group(1))
    try:
        value = int(action.action_id)
        return None if value == 7 else value
    except (TypeError, ValueError):
        return None


def _all_messages(state: GameState, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {int(m.get("index", -1)) for m in history}
    for message in state.new_messages:
        if message.index not in seen:
            history.append({"index": message.index, "sender": message.sender,
                            "recipients": list(message.recipients), "content": message.content,
                            "type": message.type, "move_index": message.move_index})
            seen.add(message.index)
    return history


class WerewolfPolicy:
    """Single registered agent with role reasoning, messaging, and safe actions."""

    def __init__(self) -> None:
        self._messages: list[dict[str, Any]] = []
        self._last_messaged_day: int | None = None
        self._logger: DecisionLogger | None = None
        self._context: DecisionContext | None = None

    def _ensure_logger(self, context: DecisionContext) -> DecisionLogger:
        if self._logger is None or self._context != context:
            self._context = context
            self._logger = DecisionLogger(context)
        return self._logger

    def choose_action(self, state: GameState, context: DecisionContext) -> LegalAction:
        if not state.legal_actions:
            raise ValueError("Werewolf decision requested without legal actions")
        game_state, players = _state_data(state)
        seat = int(context.seat_position or 0)
        role = parse_role(state.observation) or "unknown"
        known = known_roles(state.observation, seat)
        scores = build_suspicion(state.observation, game_state, players, seat, self._messages)
        phase = str(game_state.get("phase") or "")

        if phase == "night_seer":
            # Investigate the most suspicious eligible target; direct findings then
            # constrain later votes through the same private observation.
            uninvestigated = [a for a in state.legal_actions if _target(a) not in known]
            candidates = uninvestigated or state.legal_actions
            chosen = max(candidates, key=lambda a: (scores.get(_target(a), 0.0), -(_target(a) or 0)))
        elif phase == "night_wolf":
            # Remove a likely seer first; otherwise target a socially active player.
            seer_candidates = []
            for msg in self._messages:
                text = str(msg.get("content", ""))
                if re.search(r"\b(?:i am|i'm|im)\s+(?:the\s+)?seer\b|\bmy investigation\b", text, re.I):
                    seer_candidates.append(int(msg.get("sender", -1)))
            valid = state.legal_actions
            chosen = max(valid, key=lambda a: (1.5 if _target(a) in seer_candidates else 0.0,
                                               len([m for m in self._messages if int(m.get("sender", -1)) == _target(a)]),
                                               -(_target(a) or 0)))
        else:
            hard_wolves = {p for p, r in known.items() if r == "wolf"}
            candidates = [
                a for a in state.legal_actions
                if _target(a) is not None
                and (role != "wolf" or _target(a) not in hard_wolves)
            ] or state.legal_actions
            chosen = max(candidates, key=lambda a: (
                role == "seer" and _target(a) in hard_wolves,
                scores.get(_target(a), 0.0),
                -(_target(a) or 0),
            ))

        self._ensure_logger(context).decision(state, context, role, chosen, scores)
        return chosen

    def choose_message(self, state: GameState, context: DecisionContext):
        game_state, players = _state_data(state)
        _all_messages(state, self._messages)
        day = game_state.get("day")
        if self._last_messaged_day == day:
            return TERMINATE_MESSAGING
        self._last_messaged_day = day

        seat = int(context.seat_position or 0)
        role = parse_role(state.observation) or "unknown"
        known = known_roles(state.observation, seat)
        own_wolf_results = sorted(p for p, r in known.items() if r == "wolf" and p != seat)
        if role == "seer":
            latest = own_wolf_results[-1:]  # only issue positive claims that are backed by own checks
            if latest:
                content = f"I am the seer. I investigated Player{latest[0]}: wolf. I recommend voting Player{latest[0]}."
            else:
                content = "I want to compare evidence before we lock in a vote. What specific observations support your target?"
        elif role == "wolf":
            content = "Let's compare vote reasoning before deciding. A concrete explanation is more useful than a bare accusation."
        else:
            content = "Before voting, please state your strongest evidence and intended target so we can compare the claims."
        # Platform's cap is 50 words; keep our generated line comfortably below it.
        message = SendMessage(content)
        self._ensure_logger(context).message(state, context, role, message)
        return message

    def on_action_result(self, result: dict, context: DecisionContext) -> None:
        if self._logger is not None:
            self._logger.action_result(result, context)


def create_agent() -> WerewolfPolicy:
    return WerewolfPolicy()
