"""Run a stateless frozen-model service inside an allocated compute job."""

import argparse
import json
import os
from pathlib import Path

from werewolf_agent.strategy_backend import HFGenerationBackend, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Model serving must run in a Slurm allocation")
    if args.ready.exists():
        parser.error("Use a new readiness receipt")
    engine = HFGenerationBackend(args.model_path, args.model_id, args.revision,
                                 threads=args.threads)
    engine.load()
    print(json.dumps({"model_loaded": args.model_id, "revision": args.revision}), flush=True)
    serve(args.socket, engine, ready_path=args.ready)


if __name__ == "__main__":
    main()
