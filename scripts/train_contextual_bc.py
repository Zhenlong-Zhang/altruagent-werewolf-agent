"""Fit pre-PPO C0 from verified new-generator teacher journals on Slurm."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Train on a Slurm compute allocation")
    import torch
    from werewolf_agent.contextual_bc import warm_start
    from werewolf_agent.contextual_checkpoint import save_checkpoint
    from werewolf_agent.contextual_data import verify_sources
    from werewolf_agent.contextual_network import build_network
    from werewolf_agent.semantic_selector import FrozenEncoder, file_sha256
    prepared = json.loads(args.episodes.read_text())
    if prepared.get("schema") != "verified-contextual-episodes-v1" or prepared.get("mode") != "teacher":
        raise ValueError("Only verified actual-teacher data can initialize C0")
    verify_sources(prepared)
    if prepared["accounting"].get("creation_cap_exceeded"):
        raise ValueError("Resolve the recorded creation-cap violation before BC fitting")
    config = json.loads(args.config.read_text())
    if set(config) != {"epochs", "learning_rate", "dev_fraction", "seed", "chunk_size"}:
        raise ValueError("Complete explicit BC configuration required")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.manual_seed(config["seed"])
    network = build_network()
    report, optimizers = warm_start(network, FrozenEncoder(args.encoder_path), prepared["episodes"], **config)
    report.update(job=os.environ["SLURM_JOB_ID"], source_episodes_sha256=file_sha256(args.episodes),
                  config_sha256=file_sha256(args.config), collection_id=prepared["collection_id"])
    manifest = save_checkpoint(args.out, network, prepared["lineage"], parent_id=None,
                               training_report=report, optimizer_state=optimizers)
    print(json.dumps({"checkpoint_id": manifest["checkpoint_id"], "selected_epoch": report["selected_epoch"]}))


if __name__ == "__main__":
    main()
