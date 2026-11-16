"""Append-only decision logging with secrets and cross-seat data excluded."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from altruagent import DecisionContext, GameState, LegalAction


def _safe_game_state(state: GameState) -> dict[str, Any]:
    raw = state.raw if isinstance(state.raw, dict) else {}
    game = raw.get("game_state") if isinstance(raw.get("game_state"), dict) else {}
    # This is the public record; do not persist arbitrary raw API objects/tokens.
    return {
        "day": game.get("day"), "phase": game.get("phase"),
        "alive": game.get("alive", []), "dead": game.get("dead", []),
        "vote_history": game.get("vote_history", []), "winner": game.get("winner"),
    }


class DecisionLogger:
    def __init__(self, context: DecisionContext) -> None:
        base = Path(os.environ.get("WEREWOLF_LOG_DIR", "data/decision_logs")).expanduser()
        match_hash = hashlib.sha256(context.session_id.encode("utf-8")).hexdigest()[:16]
        self.path = base / f"match_{match_hash}_seat_{context.seat_position}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.match_hash = match_hash
        self.strategy_version = os.environ.get("WEREWOLF_STRATEGY_VERSION", "baseline-v0.1")
        self.role_for_eval_only: str | None = None

    def _write(self, row: dict[str, Any]) -> None:
        row.update({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "match_hash": self.match_hash,
                    "strategy_version": self.strategy_version})
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    def decision(self, state: GameState, context: DecisionContext, role: str,
                 action: LegalAction, scores: dict[int, float]) -> None:
        self.role_for_eval_only = role
        self._write({"event": "decision", "seat": context.seat_position,
                     "game_type": context.game_type, "role_for_eval_only": role,
                     "phase": state.raw.get("game_state", {}).get("phase"),
                     "observation": state.observation,
                     "public_state": _safe_game_state(state),
                     "new_messages": [m.__dict__ for m in state.new_messages],
                     "legal_actions": [{"id": a.action_id, "label": a.label} for a in state.legal_actions],
                     "chosen_action": {"id": action.action_id, "label": action.label},
                     "suspicion_scores": {str(k): round(v, 4) for k, v in scores.items()}})

    def message(self, state: GameState, context: DecisionContext, role: str, message: Any) -> None:
        self.role_for_eval_only = role
        self._write({"event": "message_decision", "seat": context.seat_position,
                     "game_type": context.game_type, "role_for_eval_only": role,
                     "public_state": _safe_game_state(state),
                     "new_messages": [m.__dict__ for m in state.new_messages],
                     "outgoing_content": message.content,
                     "outgoing_recipients": message.recipients})

    def action_result(self, result: dict[str, Any], context: DecisionContext) -> None:
        # Store only outcome metadata, never arbitrary server responses or secrets.
        state = result.get("state") if isinstance(result, dict) else None
        public: dict[str, Any] = {}
        winner = result.get("winner") if isinstance(result, dict) else None
        if isinstance(state, dict):
            game = state.get("game_state") if isinstance(state.get("game_state"), dict) else {}
            public = {"is_terminal": state.get("is_terminal"), "winner": game.get("winner"),
                      "phase": game.get("phase"), "dead": game.get("dead", [])}
            winner = winner or game.get("winner")
            if not winner and state.get("is_terminal"):
                observation = str(state.get("observation", ""))
                if "WOLVES win" in observation:
                    winner = "wolf"
                elif "VILLAGERS win" in observation:
                    winner = "villager"
        self._write({"event": "action_result", "seat": context.seat_position,
                     "role_for_eval_only": self.role_for_eval_only,
                     "error_code": result.get("error") if isinstance(result, dict) else None,
                     "winner": winner, "public_result_state": public})
