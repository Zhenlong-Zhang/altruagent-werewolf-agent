"""Compute the separate CPU probe's operation-only summary on Slurm."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.input.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        results.append({"model": path.stem, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "cases": len(rows), "valid": sum(row["valid"] for row in rows),
                        "errors": dict(Counter((row["error"] or {}).get("type", "none") for row in rows)),
                        "per_case": [{key: row[key] for key in ("scenario", "valid", "elapsed_seconds", "error")}
                                     | {"generation": row.get("completion", {}).get("provenance")} for row in rows]})
    summary = {"scope": "16-thread BF16 CPU operation-only probe; two views/model,90-second cap; not GPU screen or wins",
               "models": results}
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
