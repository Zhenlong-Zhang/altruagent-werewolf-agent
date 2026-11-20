"""Frozen contextual policy runtime; actual categorical requests, no transport."""

import copy
import hashlib
import math
import time

from .contextual_slots import RELATIONS, SERIALIZER, serialize
from .semantic_selector import ENCODER_NAME, ENCODER_REVISION
from .strategy_state import fingerprint


ENCODER_ID = f"{ENCODER_NAME}@{ENCODER_REVISION}:mean-l2-256-per-slot"


def tensorize(snapshots, encoder):
    """Batch independently encoded slots; padding never becomes a candidate."""
    import torch
    if not snapshots:
        raise ValueError("No contextual snapshots")
    for row in snapshots:
        if (row.get("serializer") != SERIALIZER
                or fingerprint({key: value for key, value in row.items() if key != "slots_hash"}) != row.get("slots_hash")):
            raise ValueError("Contextual snapshot changed after collection")
    batch, length = len(snapshots), max(len(row["slots"]) for row in snapshots)
    count = max(len(row["candidate_indices"]) for row in snapshots)
    numbers = torch.zeros(batch, length, 32)
    texts = torch.zeros(batch, length, 384)
    types = torch.zeros(batch, length, dtype=torch.int64)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    relations = torch.zeros(batch, length, length, len(RELATIONS))
    indices = torch.zeros(batch, count, dtype=torch.int64)
    legal = torch.zeros(batch, count, dtype=torch.bool)
    for index, row in enumerate(snapshots):
        size, offered = len(row["slots"]), len(row["candidate_indices"])
        if size < 9 or size > 180:
            raise ValueError("Contextual slot capacity exceeded")
        vectors = encoder.encode([slot["text"] for slot in row["slots"]])
        if vectors.shape != (size, 384) or not bool(torch.isfinite(vectors).all()):
            raise ValueError("Invalid frozen slot embeddings")
        texts[index, :size] = vectors
        numbers[index, :size] = torch.tensor([slot["numeric"] for slot in row["slots"]])
        types[index, :size] = torch.tensor([slot["type"] for slot in row["slots"]])
        mask[index, :size] = True
        for left, right, relation in row["edges"]:
            if not (0 <= left < size and 0 <= right < size and 0 <= relation < len(RELATIONS)):
                raise ValueError("Invalid typed graph edge")
            relations[index, left, right, relation] = 1.
        indices[index, :offered] = torch.tensor(row["candidate_indices"])
        legal[index, :offered] = torch.tensor(row["candidate_mask"])
    return {"text": texts, "numeric": numbers, "types": types, "mask": mask,
            "relations": relations, "candidate_indices": indices, "candidate_mask": legal}


class ContextualSelector:
    def __init__(self, network, encoder, owner, checkpoint_id, *, mode="sample", seed=97001):
        if mode not in {"sample", "greedy"} or not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("Explicit frozen checkpoint/deployment mode required")
        if (set(owner) != {"session_hash", "seat"} or type(owner["seat"]) is not int
                or owner["seat"] not in range(7) or not isinstance(owner["session_hash"], str)):
            raise ValueError("Invalid owner for contextual selector")
        self.network, self.encoder = network.eval(), encoder
        self.owner, self.checkpoint_id, self.mode = copy.deepcopy(owner), checkpoint_id, mode
        identity = hashlib.sha256(f"{seed}:{fingerprint(owner)}".encode()).hexdigest()
        self.seed_identity = identity
        self.number = 0

    def evaluate(self, snapshot):
        import torch
        from .vendor.event_ppo import masked_distribution
        if snapshot["owner"] != self.owner:
            raise ValueError("Another seat's contextual input")
        inputs = tensorize([snapshot], self.encoder)
        with torch.no_grad():
            logits, values = self.network(inputs)
            distribution = masked_distribution(logits, inputs["candidate_mask"])
        return logits[0], distribution.probs[0], values[0]

    def choose(self, packet, candidates, deadline, decision_id=None):
        started = time.monotonic()
        if packet["owner"] != self.owner:
            raise ValueError("Contextual selector cannot cross owner boundaries")
        if not math.isfinite(deadline) or started >= deadline:
            raise TimeoutError("Contextual request expired before inference; no draw")
        snapshot = serialize(packet, candidates)
        logits, tensor_probs, value = self.evaluate(snapshot)
        if time.monotonic() >= deadline:
            raise TimeoutError("Contextual inference expired before sampling; no draw")
        # Python-normalized probabilities are the exact sampler distribution.
        # PPO replay compares these to FP32 softmax within declared tolerance.
        probabilities = [float(number) for number in tensor_probs]
        total = sum(probabilities)
        probabilities = [number / total for number in probabilities]
        if not math.isfinite(float(value)):
            raise ValueError("Nonfinite behavior critic")
        self.number += 1
        decision_id = self.number if decision_id is None else decision_id
        available = [index for index, p in enumerate(probabilities) if p > 0]
        legal_count = sum(snapshot["candidate_mask"])
        uniform, draw_id = None, None
        if legal_count == 1:
            index, kind = available[0], "forced_singleton"
        elif self.mode == "greedy":
            index, kind = max(available, key=lambda i: probabilities[i]), "greedy"
        else:
            # The controller persists decision_id across reconnects. Deriving
            # each draw separately prevents a restarted worker from silently
            # replaying the beginning of an owner-local random stream.
            digest = hashlib.sha256(f"{self.seed_identity}:{decision_id}".encode()).digest()
            uniform, cumulative = (int.from_bytes(digest[:8], "big") >> 11) / 2**53, 0.
            index = available[-1]
            for current in available:
                cumulative += probabilities[current]
                if uniform < cumulative:
                    index = current
                    break
            kind = "sampled"
            draw_id = fingerprint({"owner": self.owner, "decision_id": decision_id, "draw": self.number})
        logps = [math.log(p) if p > 0 else None for p in probabilities]
        metadata = {"serialized": snapshot, "encoder_id": ENCODER_ID, "serializer_id": SERIALIZER,
                    "selection_kind": kind, "mode": self.mode, "draw_id": draw_id,
                    "sampler_uniform": uniform, "decision_id": decision_id,
                    "sampler": "owner-decision-sha256-uniform53-v1",
                    "logits": [float(number) for number in logits], "log_probabilities": logps,
                    "request_mdp": True, "elapsed_seconds": time.monotonic() - started}
        # Any subsequent deadline/race rejection is an executor outcome. The
        # actual sampled request, propensity and critic above remain immutable.
        return {"index": index, "probabilities": probabilities, "log_probability": logps[index],
                "value": float(value), "checkpoint_id": self.checkpoint_id, "metadata": metadata}
