"""Lazy, local semantic slate actor/critic with an exact legacy logit prior.

Importing this module or validating an artifact does not import Torch or load
an encoder. Neural computation belongs on a Hoffman2 compute allocation.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .selector import FEATURES, WIDTH, design_row


ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ARCHITECTURE = {
    "version": "semantic-candidate-attention-v1", "width": 32, "heads": 1,
    "embedding_dim": 384, "context_dim": 8, "residual_bound": 1.0,
    "base_dtype": "float64", "neural_dtype": "float32",
    "context_version": "coarse-self-context-v1", "max_length": 256,
    "pooling": "attention-mask-mean-l2-v1", "dropout": 0.0,
    "critic_pool_detached": True,
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_encoder(path):
    """Check staged revision/config/files without importing a model runtime."""
    path = Path(path)
    config_path = path / "config.json"
    metadata = path / ".cache/huggingface/download/config.json.metadata"
    if not config_path.is_file() or not metadata.is_file():
        raise ValueError("Semantic encoder requires local config and HF revision metadata")
    lines = metadata.read_text().splitlines()
    if not lines or lines[0] != ENCODER_REVISION:
        raise ValueError("Semantic encoder revision does not match its pinned SHA")
    if json.loads(config_path.read_text()).get("hidden_size") != 384:
        raise ValueError("Semantic encoder must have hidden_size=384")
    if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError("Semantic encoder weights must already exist locally")
    if not any((path / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
        raise ValueError("Semantic encoder tokenizer must already exist locally")
    return path


def context_inputs(features, context):
    """Use only before-decision day/phase and actor bits already in features."""
    if not features:
        raise ValueError("Semantic slate must contain candidates")
    for row in features:
        design_row(row)
    own = features[0][4:]
    if any(row[4:] != features[0][4:] for row in features):
        raise ValueError("A slate must share one acting-player context")
    alive, speech, wolf, seer = own
    if any(value not in (0, 1) for value in (speech, wolf, seer)) or wolf + seer > 1:
        raise ValueError("Invalid acting-player role/speech feature bits")
    if not 0 <= alive <= 1:
        raise ValueError("Invalid alive fraction")
    day = (context or {}).get("day", 0)
    if isinstance(day, bool) or not isinstance(day, (float, int)) or not math.isfinite(day) or day < 0:
        raise ValueError("Day must be finite and nonnegative")
    phase = "speech" if speech else (context or {}).get("phase", "unknown")
    if phase not in {"speech", "day_vote", "night_wolf", "night_seer", "unknown"}:
        raise ValueError("Unknown semantic context phase")
    values = [min(float(day), 7.) / 7., float(phase == "day_vote"),
              float(phase == "night_wolf"), float(phase == "night_seer"),
              float(speech), float(wolf), float(seer), float(alive)]
    # Serialize exactly these authorized numeric fields, never the caller's
    # other context keys (which may contain diagnostics or terminal metadata).
    text = "coarse-self-context-v1 " + json.dumps(values, separators=(",", ":"), allow_nan=False)
    return values, text


class FrozenEncoder:
    def __init__(self, path):
        self.path = validate_encoder(path)
        self._runtime = None
        self._cache = {}

    def _load(self):
        if self._runtime is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
            model = AutoModel.from_pretrained(self.path, local_files_only=True,
                                              torch_dtype=torch.float32).eval()
            model.requires_grad_(False)
            self._runtime = (tokenizer, model)
        return self._runtime

    def encode(self, texts):
        import torch
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("Semantic encoder requires nonempty text strings")
        tokenizer, model = self._load()
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        for offset in range(0, len(missing), 32):
            batch = missing[offset:offset + 32]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               max_length=ARCHITECTURE["max_length"], return_tensors="pt")
            with torch.no_grad():
                hidden = model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()
            if vectors.shape != (len(batch), 384) or not bool(torch.isfinite(vectors).all()):
                raise ValueError("Invalid semantic encoder vectors")
            for text, vector in zip(batch, vectors):
                self._cache[text] = vector.detach().clone()
        result = torch.stack([self._cache[text] for text in texts])
        # Bound persistent inference cache; keep the just-requested values.
        if len(self._cache) > 2048:
            self._cache = {text: self._cache[text] for text in dict.fromkeys(texts)}
        return result


def build_network(base_weights):
    import torch
    from torch import nn

    class SlateNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_weights = nn.Parameter(torch.tensor(base_weights, dtype=torch.float64))
            self.candidate = nn.Sequential(nn.Linear(392, 32, dtype=torch.float32), nn.Tanh())
            self.query = nn.Sequential(nn.Linear(392, 32, dtype=torch.float32), nn.Tanh())
            self.key = nn.Linear(32, 32, dtype=torch.float32)
            self.value = nn.Linear(32, 32, dtype=torch.float32)
            self.actor = nn.Sequential(nn.Linear(96, 32, dtype=torch.float32), nn.Tanh(),
                                       nn.Linear(32, 1, dtype=torch.float32))
            self.critic = nn.Sequential(nn.Linear(40, 32, dtype=torch.float32), nn.Tanh(),
                                        nn.Linear(32, 1, dtype=torch.float32))
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
            nn.init.zeros_(self.critic[-1].weight)
            nn.init.zeros_(self.critic[-1].bias)

        def forward(self, features, label_vectors, context_vector, context_embedding):
            # Keep the untouched Python float64 features for the legacy prior.
            design = torch.tensor([design_row(row) for row in features], dtype=torch.float64)
            numeric = torch.tensor(features, dtype=torch.float32)
            state = torch.tensor(context_vector, dtype=torch.float32)
            candidates = self.candidate(torch.cat((label_vectors.float(), numeric), dim=-1))
            query = self.query(torch.cat((context_embedding.float(), state), dim=-1))
            attention = torch.softmax((self.key(candidates) * query).sum(dim=-1) / math.sqrt(32), dim=0)
            pooled = (attention.unsqueeze(-1) * self.value(candidates)).sum(dim=0)
            merged = torch.cat((candidates, query.expand_as(candidates), pooled.expand_as(candidates)), dim=-1)
            residual = torch.tanh(self.actor(merged).squeeze(-1)).double()
            # Sequential multiply/add mirrors sum(design_row * old weights).
            base = torch.zeros(len(features), dtype=torch.float64)
            for index in range(WIDTH):
                base = base + design[:, index] * self.base_weights[index]
            value = self.critic(torch.cat((state, pooled.detach()), dim=-1)).squeeze(-1)
            return base + residual, value

        def actor_parameters(self):
            return [parameter for name, parameter in self.named_parameters()
                    if not name.startswith("critic.")]

    return SlateNetwork()


class SemanticSelector:
    requires_semantic_context = True

    def __init__(self, mode, weights_path, seed=None):
        if mode not in {"hybrid_collect", "hybrid_greedy"}:
            raise ValueError("Unknown semantic selector mode")
        self.mode, self.rng = mode, random.Random(seed)
        self.path = Path(weights_path)
        raw = self.path.read_bytes()
        self.record = record = json.loads(raw)
        if record.get("schema_version") != 2 or tuple(record.get("features", ())) != FEATURES:
            raise ValueError("Wrong semantic selector schema/features")
        if record.get("architecture") != ARCHITECTURE:
            raise ValueError("Unsupported semantic selector architecture")
        encoder = record.get("encoder", {})
        if not isinstance(encoder, dict) or encoder.get("name") != ENCODER_NAME or encoder.get("revision") != ENCODER_REVISION:
            raise ValueError("Unsupported semantic encoder identity")
        self.weights = tuple(record.get("base_weights", ()))
        if len(self.weights) != WIDTH or any(not isinstance(w, (float, int)) or not math.isfinite(w) or abs(w) > 8
                                             for w in self.weights):
            raise ValueError("Invalid semantic base coefficients")
        weights_file = record.get("weights_file")
        if not isinstance(weights_file, str) or not weights_file:
            raise ValueError("Semantic checkpoint requires a tensor filename")
        self.tensor_path = self.path.parent / weights_file
        actual_hash = file_sha256(self.tensor_path)
        if actual_hash != record.get("weights_sha256"):
            raise ValueError("Semantic tensor checkpoint checksum mismatch")
        encoder_path = os.environ.get("WEREWOLF_SEMANTIC_ENCODER", encoder.get("model_path"))
        if not isinstance(encoder_path, str) or not encoder_path:
            raise ValueError("Semantic checkpoint requires a local encoder path")
        self.encoder_path = validate_encoder(encoder_path)
        for filename, expected_hash in encoder.get("files_sha256", {}).items():
            if file_sha256(self.encoder_path / filename) != expected_hash:
                raise ValueError("Semantic encoder staged-file checksum mismatch")
        self.checkpoint_id = hashlib.sha256(raw + actual_hash.encode()).hexdigest()[:16]
        self._network = None
        self._encoder = None

    def _load(self):
        if self._network is None:
            import torch
            network = build_network(self.weights)
            state = torch.load(self.tensor_path, map_location="cpu", weights_only=True)
            expected = network.state_dict()
            if (not isinstance(state, dict) or state.keys() != expected.keys()
                    or any(not isinstance(state[key], torch.Tensor) or state[key].dtype != value.dtype
                           or state[key].shape != value.shape for key, value in expected.items())):
                raise ValueError("Semantic tensor keys, shapes or dtypes do not match architecture")
            network.load_state_dict(state, strict=True)
            if not all(bool(torch.isfinite(value).all()) for value in network.state_dict().values()):
                raise ValueError("Nonfinite semantic checkpoint weights")
            actual_base = network.base_weights.detach().tolist()
            if actual_base != list(self.weights):
                raise ValueError("Semantic metadata/tensor base coefficients differ")
            self._network = network.eval()
            self._encoder = FrozenEncoder(self.encoder_path)

    def probabilities(self, features, labels, context, *, deadline=None):
        import torch
        if deadline is not None and (not isinstance(deadline, (float, int)) or not math.isfinite(deadline)):
            raise ValueError("Semantic deadline must be a finite monotonic timestamp")

        def check_deadline():
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Semantic selector exceeded its soft compute budget")

        check_deadline()
        if len(features) != len(labels) or not features:
            raise ValueError("Semantic features and labels must have equal nonzero lengths")
        values, text = context_inputs(features, context)
        self._load()
        check_deadline()
        vectors = self._encoder.encode([*labels, text])
        check_deadline()
        with torch.no_grad():
            logits, _ = self._network(features, vectors[:-1], values, vectors[-1])
            mass = torch.softmax(logits, dim=0)
        check_deadline()
        if not bool(torch.isfinite(mass).all()) or not math.isclose(float(mass.sum()), 1., abs_tol=1e-12):
            raise ValueError("Invalid semantic policy probabilities")
        result = mass.tolist()
        check_deadline()
        return result

    def choose(self, features, labels, context, *, deadline=None):
        if len(features) < 2:
            raise ValueError("Semantic selection requires distinct options")
        mass = self.probabilities(features, labels, context, deadline=deadline)
        if self.mode == "hybrid_collect":
            return self.rng.choices(range(len(mass)), weights=mass)[0], mass
        choice = max(range(len(mass)), key=mass.__getitem__)
        return choice, [float(index == choice) for index in range(len(mass))]
