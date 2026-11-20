"""Explicit, source-pinned teacher/learned deployment factory.

WEREWOLF_CONTEXTUAL_CONFIG names a contextual-runtime-v1 JSON file. Every
field is required; node-local paths may change only through a new config hash.
The source manifest is a files{relative_path:sha256} JSON object. The SDK
manifest is the isolated strategy-sdk-overlay-v2 builder's output.
"""

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from .strategy_backend import GenerationClient, LocalTokenCounter
from .strategy_entry import StrategyTable
from .strategy_state import fingerprint
from .contextual_selector import ENCODER_ID
from .contextual_slots import SERIALIZER


class GeneratorProvenanceError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def load_config(path):
    config = json.loads(Path(path).read_text())
    fields = {"schema", "collection_id", "protocol_hash", "mode", "checkpoint_id", "lineage", "source_manifest_sha256",
              "sdk_manifest_sha256", "generator_spec", "paths", "seed"}
    if not isinstance(config, dict) or set(config) != fields or config["schema"] != "contextual-runtime-v1":
        raise ValueError("Explicit contextual runtime schema required")
    if (not isinstance(config["collection_id"], str) or not config["collection_id"] or config["mode"] not in {"teacher", "sample", "greedy"}
            or not all(digest(config[key]) for key in ("protocol_hash", "source_manifest_sha256", "sdk_manifest_sha256"))
            or not isinstance(config["lineage"], dict) or not config["lineage"]
            or type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32):
        raise ValueError("Invalid runtime lineage, mode or collection identity")
    paths = config["paths"]
    if not isinstance(paths, dict) or set(paths) != {"socket", "tokenizer", "encoder", "journal_dir", "checkpoint", "source_manifest", "sdk_manifest"}:
        raise ValueError("Explicit runtime paths required")
    for key, value in paths.items():
        if key in {"encoder", "checkpoint"} and config["mode"] == "teacher" and value is None:
            continue
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise ValueError("Runtime paths must be explicit absolute paths")
    if (config["mode"] == "teacher") != (config["checkpoint_id"] is None):
        raise ValueError("Teacher mode has no contextual checkpoint; learned mode pins one")
    if config["mode"] != "teacher" and not digest(config["checkpoint_id"]):
        raise ValueError("Learned runtime requires a full checkpoint hash")
    spec = config["generator_spec"]
    if not isinstance(spec, dict) or set(spec) != {"model_id", "revision", "backend_version", "dtype", "device", "max_new_tokens", "temperature", "model_seconds", "input_tokens"}:
        raise ValueError("Complete generator specification required")
    if (not isinstance(spec["model_id"], str) or not spec["model_id"] or not isinstance(spec["backend_version"], str) or not spec["backend_version"]
            or not isinstance(spec["revision"], str) or not re.fullmatch(r"[a-f0-9]{40}", spec["revision"])
            or spec["dtype"] not in {"torch.bfloat16", "torch.float16", "torch.float32"} or spec["device"] not in {"cpu", "cuda"}
            or type(spec["max_new_tokens"]) is not int or not 1 <= spec["max_new_tokens"] <= 2048
            or type(spec["input_tokens"]) is not int or not 1 <= spec["input_tokens"] <= 8192):
        raise ValueError("Invalid pinned model/decode configuration")
    for key, low, high in (("temperature", 0, 1.5), ("model_seconds", 0, 30)):
        value = spec[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high or key == "model_seconds" and value == 0:
            raise ValueError("Invalid model budget/temperature")
    lineage = config["lineage"]
    if (set(lineage) != {"rules", "source_manifest", "generator", "controller", "realizer", "encoder", "serializer", "population"}
            or any(not isinstance(value, str) or not value.strip() for value in lineage.values())
            or lineage["generator"] != fingerprint(spec)
            or lineage["source_manifest"] != config["source_manifest_sha256"]
            or lineage["encoder"] != ENCODER_ID or lineage["serializer"] != SERIALIZER
            or lineage["rules"] != "altruagent-seven-fixed-roles-v1" or lineage["realizer"] != "typed-canonical-v1"):
        raise ValueError("Runtime lineage does not bind its exact source/generator/encoder/serializer")
    json.dumps(config, allow_nan=False)
    return config


def verify_sources(config):
    root = Path(__file__).resolve().parents[1]
    for key, hash_key in (("source_manifest", "source_manifest_sha256"), ("sdk_manifest", "sdk_manifest_sha256")):
        if sha(config["paths"][key]) != config[hash_key]:
            raise ValueError("Runtime manifest bytes changed")
    manifest = json.loads(Path(config["paths"]["source_manifest"]).read_text())
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Source manifest must enumerate its frozen files")
    # Pin the complete small policy package, including lazy/vendor imports.
    # Otherwise an omitted helper can silently change a supposedly fixed actor.
    required = {str(path.relative_to(root)) for path in (root / "werewolf_agent").rglob("*.py")}
    if not required.issubset(manifest["files"]):
        raise ValueError("Source manifest omits an active runtime module")
    for name, expected in manifest["files"].items():
        if not isinstance(name, str):
            raise ValueError("Source paths must be strings")
        target = root / name
        if Path(name).is_absolute() or ".." in Path(name).parts or target.is_symlink() or not digest(expected):
            raise ValueError("Unsafe source manifest entry")
        if not target.resolve().is_relative_to(root) or sha(target) != expected:
            raise ValueError("Runtime source file changed")
    sdk_path = Path(config["paths"]["sdk_manifest"])
    sdk = json.loads(sdk_path.read_text())
    if (sdk.get("schema") != "strategy-sdk-overlay-v2" or not isinstance(sdk.get("files"), dict)
            or "altruagent/runner.py" not in sdk["files"]):
        raise ValueError("Source-pinned isolated SDK required")
    import altruagent.runner
    if Path(altruagent.runner.__file__).resolve() != (sdk_path.parent / "altruagent/runner.py").resolve():
        raise ValueError("Loaded SDK is not the declared isolated overlay")
    for name, hashes in sdk["files"].items():
        if not isinstance(name, str):
            raise ValueError("SDK paths must be strings")
        target = sdk_path.parent / name
        if Path(name).is_absolute() or ".." in Path(name).parts or target.is_symlink() or not target.resolve().is_relative_to(sdk_path.parent.resolve()):
            raise ValueError("Unsafe SDK manifest entry")
        if not isinstance(hashes, dict) or not digest(hashes.get("overlay_sha256")) or sha(target) != hashes["overlay_sha256"]:
            raise ValueError("Loaded SDK source changed")


class PinnedBackend:
    def __init__(self, backend, counter, spec):
        self.backend, self.counter, self.spec = backend, counter, copy.deepcopy(spec)

    def complete(self, messages, max_new_tokens, temperature, seed, deadline):
        spec = self.spec
        if max_new_tokens != spec["max_new_tokens"] or temperature != spec["temperature"]:
            raise ValueError("Unversioned decode settings")
        remaining = deadline - time.monotonic()
        if remaining <= 0 or remaining > spec["model_seconds"] + .01:
            raise ValueError("Unexpected model deadline budget")
        count = self.counter(messages)
        if count > spec["input_tokens"]:
            raise ValueError("Native input exceeded declared cap")
        result = self.backend.complete(messages, max_new_tokens, temperature, seed, deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError("Pinned backend returned after the model deadline")
        provenance = result["provenance"]
        expected = {"model_id": spec["model_id"], "revision": spec["revision"], "backend_version": spec["backend_version"],
                    "actual_dtype": spec["dtype"], "device": spec["device"], "max_new_tokens": max_new_tokens,
                    "temperature": temperature, "seed": seed, "max_input_tokens": spec["input_tokens"], "input_tokens": count, "thinking_enabled": True}
        if any(type(provenance.get(key)) is not type(value) or provenance.get(key) != value for key, value in expected.items()):
            raise GeneratorProvenanceError("Generator provenance differs from the pinned runtime")
        return result


def create_agent(config_path=None):
    config = load_config(config_path or os.environ["WEREWOLF_CONTEXTUAL_CONFIG"])
    verify_sources(config)
    paths, spec = config["paths"], config["generator_spec"]
    counter = LocalTokenCounter(paths["tokenizer"], spec["revision"])
    backend = PinnedBackend(GenerationClient(paths["socket"]), counter, spec)
    selector_factory = None
    if config["mode"] != "teacher":
        import torch
        torch.set_num_threads(1)  # Multiple SDK workers share one allocation.
        from .contextual_checkpoint import load_checkpoint
        from .contextual_selector import ContextualSelector
        from .semantic_selector import FrozenEncoder
        network, manifest = load_checkpoint(paths["checkpoint"])
        if (manifest["checkpoint_id"] != config["checkpoint_id"]
                or set(manifest["lineage"]) != set(config["lineage"])
                or any(value != config["lineage"][key] for key, value in manifest["lineage"].items() if key != "population")):
            raise ValueError("Contextual checkpoint differs from declared runtime lineage")
        encoder = FrozenEncoder(paths["encoder"])
        def selector_factory(context):
            owner = {"session_hash": hashlib.sha256(context.session_id.encode()).hexdigest()[:16], "seat": context.seat_position}
            return ContextualSelector(network, encoder, owner, manifest["checkpoint_id"], mode=config["mode"], seed=config["seed"])
    metadata = {key: copy.deepcopy(config[key]) for key in ("collection_id", "protocol_hash", "mode", "checkpoint_id", "lineage", "generator_spec", "source_manifest_sha256", "sdk_manifest_sha256")}
    metadata.update(runtime_config_hash=fingerprint(config), source_verified=True)
    return StrategyTable(backend, counter, paths["journal_dir"], selector_factory=selector_factory,
        policy_options={"model_seconds": spec["model_seconds"], "max_new_tokens": spec["max_new_tokens"],
                        "temperature": spec["temperature"], "input_budget": spec["input_tokens"], "seed": config["seed"]},
        runtime_metadata=metadata)
