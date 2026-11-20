"""Train one verified fresh batch on Slurm; never launch games or infer labels."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Training belongs on a Slurm compute allocation")
    import torch
    from werewolf_agent.contextual_checkpoint import load_checkpoint, save_checkpoint
    from werewolf_agent.contextual_ppo import ContextualPPO
    from werewolf_agent.semantic_selector import FrozenEncoder, file_sha256
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    network, manifest = load_checkpoint(args.input_checkpoint)
    lineage = json.loads(args.lineage.read_text())
    if any(lineage.get(key) != value for key, value in manifest["lineage"].items() if key != "population"):
        raise ValueError("Changed generator/controller/serializer requires a new training lineage")
    trainer = ContextualPPO(network, FrozenEncoder(args.encoder_path), manifest["checkpoint_id"],
                             lineage=lineage, config=json.loads(args.config.read_text()))
    if "optimizer_file" in manifest:
        state = torch.load(args.input_checkpoint / manifest["optimizer_file"], map_location="cpu", weights_only=True)
        trainer.actor_optimizer.load_state_dict(state["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for optimizer in (trainer.actor_optimizer, trainer.critic_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = trainer.config["learning_rate"]
    prepared = json.loads(args.episodes.read_text())
    if (prepared.get("schema") != "verified-contextual-episodes-v1" or prepared.get("lineage") != lineage
            or prepared.get("mode") != "sample"):
        raise ValueError("Use the verified fresh-journal collector adapter before training")
    from werewolf_agent.contextual_data import verify_sources
    verify_sources(prepared)
    if prepared["accounting"].get("creation_cap_exceeded"):
        raise ValueError("Preserve all created attempts and resolve the protocol violation before training")
    report = trainer.update(prepared["episodes"], chunk_size=args.chunk_size)
    report.update(job=os.environ["SLURM_JOB_ID"], source_episodes_sha256=file_sha256(args.episodes),
                  optimizer_config_sha256=file_sha256(args.config), chunk_size=args.chunk_size,
                  lineage=lineage, parent_checkpoint=manifest["checkpoint_id"],
                  scope="PPO request-policy weights only; frozen language model and encoder unchanged")
    result = save_checkpoint(args.out, network, lineage, parent_id=manifest["checkpoint_id"], training_report=report,
        optimizer_state={"actor_optimizer": trainer.actor_optimizer.state_dict(),
                         "critic_optimizer": trainer.critic_optimizer.state_dict()})
    print(json.dumps({"checkpoint_id": result["checkpoint_id"], "accepted_steps": report["accepted_steps"],
                      "rejected_steps": report["rejected_steps"]}))


if __name__ == "__main__":
    main()
