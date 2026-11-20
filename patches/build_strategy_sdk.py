"""Copy and patch a pinned local SDK; never mutate the source installation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PACKAGE_FILES = (
    "__init__.py", "_responses.py", "agent_loader.py", "auth.py", "client.py", "console.py",
    "errors.py", "game.py", "mcp_game.py", "mcp_transport.py", "models.py", "notices.py",
    "official.py", "runner.py", "runtime.py", "supervisor.py", "worker.py",
)
ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE", "LICENSE.txt", "LICENSE.md")


def build(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("SDK overlay destination must be new")
    if (source / "altruagent").is_symlink():
        raise ValueError("SDK package must not be a symbolic link")
    files = [Path("altruagent") / name for name in PACKAGE_FILES]
    files += [Path(name) for name in ROOT_FILES if (source / name).exists()]
    for relative in files:
        if (source / relative).is_symlink() or not (source / relative).is_file():
            raise ValueError("SDK allowlist requires regular source files")
    destination.mkdir(parents=True)
    source_hashes = {}
    for relative in files:
        original_file = source / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_file, target)
        source_hashes[str(relative)] = hashlib.sha256(original_file.read_bytes()).hexdigest()
    runner = destination / "altruagent" / "runner.py"
    original = runner.read_text()
    text = original

    def change(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("SDK source does not match the reviewed hook anchor")
        text = text.replace(old, new)

    change('    state = read(game.get_state)\n\n    while True:\n',
           '    from werewolf_agent import strategy_transport as _strategy_transport\n'
           '    state = read(game.get_state)\n'
           '    _strategy, context = _strategy_transport.resolve(choose_action, state, context)\n\n'
           '    while True:\n'
           '        _strategy_transport.observe(_strategy, state, context, now)\n')
    change('        if state.phase == _MESSAGING_PHASE:\n            decision =',
           '        if state.phase == _MESSAGING_PHASE:\n'
           '            if _strategy is not None and _strategy.memory.current.terminated:\n'
           '                state = wait(state, _last_message_seq(state))\n'
           '                continue\n'
           '            if _strategy is not None and _strategy.memory.current.pending is not None:\n'
           '                _strategy_transport.refresh(_strategy, game, state, context, now)\n'
           '            decision =')
    change('            try:\n                if decision is TERMINATE_MESSAGING:',
           '            decision, _strategy_delivery_id = _strategy_transport.prepare(_strategy, decision, context, now)\n'
           '            try:\n                if decision is TERMINATE_MESSAGING:')
    change('                _report_result(message_result_hook, {"error": code or "MESSAGE_DELIVERY_UNKNOWN"}, context)',
           '                _strategy_transport.receipt(_strategy, _strategy_delivery_id, {"error": code or "MESSAGE_DELIVERY_UNKNOWN"}, state, context, now)\n'
           '                _report_result(message_result_hook, {"error": code or "MESSAGE_DELIVERY_UNKNOWN"}, context)')
    change('            _report_result(message_result_hook, result, context)',
           '            _strategy_transport.receipt(_strategy, _strategy_delivery_id, result, state, context, now)\n'
           '            _report_result(message_result_hook, result, context)')
    change('        if state.is_current_actor:\n            realtime =',
           '        if state.is_current_actor:\n'
           '            if _strategy is not None:\n'
           '                _strategy_transport.refresh(_strategy, game, state, context, now)\n'
           '            realtime =')
    runner.write_text(text)
    manifest = {"schema": "strategy-sdk-overlay-v2", "copy_policy": "explicit-sdk-source-allowlist-v1",
                "source_runner_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "overlay_runner_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": {str(relative): {"source_sha256": source_hashes[str(relative)],
                    "overlay_sha256": hashlib.sha256((destination / relative).read_bytes()).hexdigest()}
                    for relative in files}}
    (destination / "strategy-overlay.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination)))
