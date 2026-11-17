"""Run one synthetic voting decision through each actual LLM policy."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from altruagent import DecisionContext, GameState

from werewolf_agent.variants import LLMPolicy


def main():
    context = DecisionContext("model-smoke", None, "werewolf", "smoke", 0)
    names = [f"Player{i}" for i in range(7)]
    state = GameState.from_mcp_state({
        "session_id": "model-smoke",
        "game_type": "werewolf",
        "observation": "Your role: SEER\nYour investigations: Player2 is wolf",
        "game_state": {"day": 1, "phase": "day_vote", "alive": names},
        "players": [{"position": i, "name": name} for i, name in enumerate(names)],
        "legal_actions": {"actions": [
            {"action_id": str(seat), "label": f"Vote to eliminate Player{seat}"}
            for seat in (2, 5)
        ]},
    })
    with TemporaryDirectory() as log_dir:
        os.environ["WEREWOLF_LOG_DIR"] = log_dir
        for variant in ("direct", "belief"):
            policy = LLMPolicy(variant)
            try:
                choice = policy.choose_action(state, context)
                path = next(Path(log_dir).glob("*.jsonl"))
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                model_call = next(row for row in rows if row["event"] == "model_call")
                print(variant, choice.action_id, model_call["elapsed_seconds"],
                      "fallback", model_call["used_fallback"], flush=True)
                assert choice.action_id in {"2", "5"} and not model_call["used_fallback"]
                path.unlink()
            finally:
                policy.client.close()


if __name__ == "__main__":
    main()
