"""Stochastic policy over a shared LLM candidate slate."""

import hashlib
import json
import math
from pathlib import Path
import random

FEATURES = (
    "llm_score", "rule_score", "public_support", "known_alignment",
    "alive_fraction", "is_speech", "is_wolf", "is_seer",
)
# Context-only terms cancel in a softmax. Interact context with each
# candidate feature so the value of an action can depend on role and phase.
WIDTH = 4 * 5
INITIAL_WEIGHTS = (1.5, 0.5, 0.0, 0.0) + (0.0,) * (WIDTH - 4)


def design_row(row):
    if len(row) != len(FEATURES) or any(not math.isfinite(x) or abs(x) > 1 for x in row):
        raise ValueError("Expected normalized, finite candidate features")
    return [value * condition for condition in (1, *row[4:]) for value in row[:4]]


def probabilities(rows, weights):
    logits = [sum(x*w for x, w in zip(design_row(row), weights)) for row in rows]
    peak = max(logits)
    mass = [math.exp(value - peak) for value in logits]
    total = sum(mass)
    return [value / total for value in mass]


class Selector:
    def __init__(self, mode, weights_path=None, seed=None):
        if mode not in {"hybrid_collect", "hybrid_greedy"}:
            raise ValueError("Unknown hybrid selector mode")
        self.mode = mode
        self.weights = INITIAL_WEIGHTS
        if weights_path is not None:
            record = json.loads(Path(weights_path).read_text())
            if record.get("schema_version") != 1 or tuple(record["features"]) != FEATURES:
                raise ValueError("Wrong hybrid checkpoint schema")
            self.weights = tuple(record["weights"])
            if len(self.weights) != WIDTH or any(not math.isfinite(w) or abs(w) > 8 for w in self.weights):
                raise ValueError("Invalid selector weights")
        payload = json.dumps(list(self.weights), separators=(",", ":")).encode()
        self.checkpoint_id = hashlib.sha256(payload).hexdigest()[:16]
        self.rng = random.Random(seed)

    def choose(self, features):
        if len(features) < 2:
            raise ValueError("Hybrid selection requires distinct options")
        mass = probabilities(features, self.weights)
        if self.mode == "hybrid_collect":
            return self.rng.choices(range(len(mass)), weights=mass)[0], mass
        choice = max(range(len(mass)), key=mass.__getitem__)
        return choice, [float(i == choice) for i in range(len(mass))]


def load_selector(mode, weights_path=None, seed=None):
    """Dispatch checkpoint formats without importing a neural runtime for legacy use."""
    if weights_path is not None:
        record = json.loads(Path(weights_path).read_text())
        if record.get("schema_version") == 2:
            from .semantic_selector import SemanticSelector
            return SemanticSelector(mode, weights_path, seed)
    return Selector(mode, weights_path, seed)
