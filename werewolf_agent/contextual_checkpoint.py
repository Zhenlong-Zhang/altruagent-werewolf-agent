"""Hash-bound contextual checkpoint bundles; tensors use safetensors only."""

import json
from pathlib import Path

from .contextual_network import ARCHITECTURE, build_network
from .contextual_selector import ENCODER_ID
from .contextual_slots import SERIALIZER
from .semantic_selector import file_sha256
from .strategy_state import fingerprint


def save_checkpoint(directory, network, lineage, *, parent_id, training_report, optimizer_state=None):
    from safetensors.torch import save_file
    import torch
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Checkpoint outputs must be new immutable directories")
    directory.mkdir(parents=True, mode=0o700)
    weights = directory / "actor_critic.safetensors"
    save_file({name: value.detach().cpu().contiguous() for name, value in network.state_dict().items()}, str(weights))
    report_path = directory / "training_report.json"
    report_path.write_text(json.dumps(training_report, indent=2, allow_nan=False) + "\n")
    manifest = {"schema": "contextual-checkpoint-v1", "architecture": ARCHITECTURE,
                "encoder_id": ENCODER_ID, "serializer_id": SERIALIZER,
                "weights_file": weights.name, "weights_sha256": file_sha256(weights),
                "report_file": report_path.name, "report_sha256": file_sha256(report_path),
                "parent_id": parent_id, "lineage": lineage, "torch_version": str(torch.__version__)}
    if optimizer_state is not None:
        optimizer_file = directory / "optimizers.pt"
        torch.save(optimizer_state, optimizer_file)
        manifest.update(optimizer_file=optimizer_file.name, optimizer_sha256=file_sha256(optimizer_file))
    manifest["checkpoint_id"] = fingerprint(manifest)
    (directory / "checkpoint.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_checkpoint(directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    manifest = json.loads((directory / "checkpoint.json").read_text())
    if (manifest.get("schema") != "contextual-checkpoint-v1" or manifest.get("architecture") != ARCHITECTURE
            or manifest.get("encoder_id") != ENCODER_ID or manifest.get("serializer_id") != SERIALIZER
            or manifest.get("weights_file") != "actor_critic.safetensors"
            or manifest.get("report_file") != "training_report.json"
            or fingerprint({key: value for key, value in manifest.items() if key != "checkpoint_id"}) != manifest.get("checkpoint_id")):
        raise ValueError("Checkpoint identity/schema mismatch")
    for prefix in ("weights", "report"):
        if file_sha256(directory / manifest[f"{prefix}_file"]) != manifest[f"{prefix}_sha256"]:
            raise ValueError("Checkpoint artifact changed")
    if "optimizer_file" in manifest:
        if (manifest["optimizer_file"] != "optimizers.pt"
                or file_sha256(directory / manifest["optimizer_file"]) != manifest["optimizer_sha256"]):
            raise ValueError("Checkpoint optimizer state changed")
    network = build_network()
    network.load_state_dict(load_file(str(directory / manifest["weights_file"])), strict=True)
    return network.eval(), manifest
