"""Verify the complete registered collection; never turn missing games into losses."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Prepare model data on a Slurm compute allocation")
    from werewolf_agent.contextual_data import prepare_batch
    if args.out.exists():
        parser.error("Never replace a prepared batch")
    result = prepare_batch(args.protocol, args.registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result["accounting"]))


if __name__ == "__main__":
    main()
