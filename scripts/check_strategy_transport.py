"""Execute actual isolated SDK runner, including its final terminate payload."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import hashlib
import sys
import tempfile
from types import SimpleNamespace


def exercise(hooked, refresh_fails=False, mixed=False):
    from altruagent import DecisionContext, TERMINATE_MESSAGING
    from altruagent.models import GameState
    from altruagent.runner import run_game
    from werewolf_agent.strategy_memory import StrategyMemory
    import hashlib

    with tempfile.TemporaryDirectory() as directory:
        context = DecisionContext(session_id="transport-fixture", tournament_id="fixture", game_type="werewolf", agent_id="self", seat_position=6)
        owner = {"session_hash": hashlib.sha256(context.session_id.encode()).hexdigest()[:16], "seat": 6}
        memory = StrategyMemory(owner, Path(directory) / "private.jsonl")
        last = {"seq": 9, "sender": 1, "recipients": [], "type": "chat", "content": "Player6 please explain your vote.", "day": 2}

        class Game:
            phase, refreshes, chats = "messaging", 0, 0

            def state(self):
                return GameState.from_mcp_state({"session_id": context.session_id, "game_type": "werewolf",
                    "observation": "Your role: VILLAGER", "phase": self.phase, "is_terminal": self.phase == "terminal",
                    "is_current_actor": self.phase == "moving", "state_version": 1,
                    "messaging_seconds_left": 50, "new_messages": [],
                    "players": [{"position": i, "name": f"Player{i}"} for i in range(7)],
                    "game_state": {"day": 2, "phase": "day_vote", "alive": [f"Player{i}" for i in range(7)], "dead": [], "vote_history": []},
                    "legal_actions": {"actions": [{"action_id": "opaque:vote", "label": "Vote Player1", "input": {}}]}})

            get_state = state

            def send_message(self, **kwargs):
                assert kwargs["message_type"] == "terminate"
                self.chats += 1
                self.phase = "moving"
                return {"accepted": True, "phase": "moving", "new_messages": [last]}

            def get_messages(self, **kwargs):
                self.refreshes += 1
                if refresh_fails:
                    raise OSError("fixture refresh failure")
                return {"messages": [last]}

            def play_action(self, **kwargs):
                self.phase = "terminal"
                return self.state()

            def get_result(self):
                return {"is_terminal": True, "returns": {"self": 1}, "your_return": 1}

        class Policy:
            requires_visible_transcript_flush = hooked

            def __init__(self):
                self.memory = memory
                self.packets = []

            def on_visible_update(self, state, context, envelopes, source, complete, observed_at):
                memory.observe_state(state, context, observed_at)
                if source == "refresh_error":
                    memory.mark_flush_error("authorized_refresh_failed")
                else:
                    memory.merge_visible(envelopes, source, complete, observed_at)

            def choose_message(self, state, context):
                return TERMINATE_MESSAGING

            def before_message_request(self, *args):
                pass

            def on_delivery_result(self, identifier, result, context, observed_at):
                memory.terminate(result.get("accepted") is True)

            def choose_action(self, state, context):
                if hooked:
                    self.packets.append(memory.compile_packet(state, context, lambda _: 100))
                return state.legal_actions[0]

        game, policy = Game(), Policy()
        agent = policy
        if mixed:
            class MixedRouter:
                requires_visible_transcript_flush = True

                def strategy_transport_policy(self, context):
                    return policy

                def choose_message(self, state, context):
                    return policy.choose_message(state, context)

                def choose_action(self, state, context):
                    return policy.choose_action(state, context)
            agent = MixedRouter()
        try:
            final = run_game(game, context, agent, now=lambda: 10., sleep=lambda _: None, log=lambda _: None)
            assert final.is_terminal
            assert game.chats == 1
            if hooked:
                packet = policy.packets[0]
                assert [row["message_ref"] for row in packet["attributed_claims"]] == ["message:9"]
                assert packet["window"]["transcript_complete"] is (not refresh_fails)
                assert len(memory.messages) == 1
                assert game.refreshes == 1
            else:
                assert game.refreshes == 0
        finally:
            memory.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        exercise(True)
        exercise(True, True)
        exercise(False)
        exercise(False, mixed=True)
        print(json.dumps({"actual_runner_cases": 4, "passed": True}))
        return
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("strategy_sdk_builder", root / "patches/build_strategy_sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as directory:
        fixture_source = Path(directory) / "source"
        (fixture_source / "altruagent").mkdir(parents=True)
        for name in module.PACKAGE_FILES:
            shutil.copyfile(root / "starter" / "altruagent" / name, fixture_source / "altruagent" / name)
        for relative in (".env", ".git/config", ".venv/runtime.py", "altruagent/extra_runtime.py"):
            target = fixture_source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic excluded fixture\n")
        check_overlay = Path(directory) / "checked-allowlist"
        manifest = module.build(fixture_source, check_overlay)
        assert set(manifest["files"]) == {f"altruagent/{name}" for name in module.PACKAGE_FILES}
        assert not any((check_overlay / path).exists() for path in (".env", ".git", ".venv", "altruagent/extra_runtime.py"))
        for relative, hashes in manifest["files"].items():
            assert hashes["source_sha256"] == hashlib.sha256((fixture_source / relative).read_bytes()).hexdigest()
            assert hashes["overlay_sha256"] == hashlib.sha256((check_overlay / relative).read_bytes()).hexdigest()
        overlay = Path(directory) / "overlay"
        module.build(root / "starter", overlay)
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(overlay), str(root), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])]))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child"], env=environment, check=True)


if __name__ == "__main__":
    main()
