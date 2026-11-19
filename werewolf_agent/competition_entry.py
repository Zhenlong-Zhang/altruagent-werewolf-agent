"""One configured hybrid competitor at every seat assigned by the official runtime."""

import os
from pathlib import Path

from .hybrid import HybridPolicy
from .variants import LLMPolicy


def create_agent():
    mode = os.environ.get("WEREWOLF_HYBRID_MODE", "hybrid_greedy")
    if mode not in {"llm_select", "hybrid_greedy", "hybrid_collect"}:
        raise ValueError("Select llm_select, hybrid_greedy or hybrid_collect explicitly for competition")
    backend = os.environ.get("WEREWOLF_MODEL_PATH") or os.environ.get("WEREWOLF_CPU_MODEL")
    if not backend or not (Path(backend) / "config.json").is_file():
        raise ValueError("Competition requires a staged Hugging Face model directory with config.json")
    weights = os.environ.get("WEREWOLF_HYBRID_WEIGHTS")
    if mode != "llm_select" and not weights:
        raise ValueError("Competition selector modes require an explicit WEREWOLF_HYBRID_WEIGHTS checkpoint")
    policy = HybridPolicy(mode, weights, allow_tournament=True)
    policy._model = LLMPolicy("hybrid")
    identity = policy.selector.checkpoint_id if policy.selector is not None else "llm_direct"
    policy._strategy_version = f"competition:{mode}:{identity}"
    return policy
