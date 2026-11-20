"""Stateless local generation for strategy agents; shared weights, private requests.

Model imports/loading are lazy and belong on a Slurm compute node. The Unix
socket service avoids one language-model copy per contestant worker. It never
retains dialogue, KV state, prompts or generated reasoning between requests.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import time

import httpx


BACKEND_VERSION = "strategy-native-generation-v1"
MAX_REQUEST_BYTES = 262144


class BackendError(RuntimeError):
    pass


def validate_request(messages, max_new_tokens, temperature, seed, deadline):
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise BackendError("Expected one to eight complete chat messages")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str) or not message["content"]):
            raise BackendError("Invalid stateless message envelope")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 2048:
        raise BackendError("Invalid completion token budget")
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 1.5):
        raise BackendError("Invalid sampling temperature")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise BackendError("Invalid request seed")
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline) or deadline <= time.monotonic()):
        raise BackendError("Request expired before generation")
    if deadline - time.monotonic() > 120:
        raise BackendError("Request deadline exceeds supported window")


class GenerationClient:
    """Owner-only UDS client; requests include all of their own context."""

    def __init__(self, socket_path):
        self.path = Path(socket_path)
        if (self.path.parent.stat().st_uid != os.getuid()
                or self.path.parent.stat().st_mode & 0o077):
            raise BackendError("Service directory must be private to this user")
        transport = httpx.HTTPTransport(uds=str(self.path), retries=0)
        self.client = httpx.Client(transport=transport, trust_env=False)

    def complete(self, messages, max_new_tokens, temperature, seed, deadline):
        validate_request(messages, max_new_tokens, temperature, seed, deadline)
        payload = {"messages": messages, "max_new_tokens": max_new_tokens,
                   "temperature": temperature, "seed": seed, "deadline": deadline}
        try:
            result = self.client.post("http://strategy.local/complete", json=payload,
                                      timeout=max(.001, deadline - time.monotonic()))
            result.raise_for_status()
            answer = result.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendError(type(exc).__name__) from exc
        if time.monotonic() > deadline:
            raise BackendError("Late service response rejected")
        if (not isinstance(answer, dict) or not isinstance(answer.get("text"), str)
                or not isinstance(answer.get("provenance"), dict)):
            raise BackendError("Invalid service response")
        return answer

    def close(self):
        self.client.close()


class HFGenerationBackend:
    """Native Qwen3 reasoning completion, with no inherited chat/KV cache."""

    def __init__(self, model_path, model_id, revision, *, device="cuda",
                 dtype="bfloat16", threads=4, max_input_tokens=8192):
        self.path, self.model_id, self.revision = Path(model_path), model_id, revision
        if device not in {"cpu", "cuda"} or dtype not in {"float32", "float16", "bfloat16"}:
            raise BackendError("Unsupported explicit runtime")
        if type(threads) is not int or threads < 1:
            raise BackendError("Invalid CPU thread count")
        if type(max_input_tokens) is not int or not 512 <= max_input_tokens <= 16384:
            raise BackendError("Invalid input budget")
        metadata = self.path / ".cache/huggingface/download/config.json.metadata"
        lines = metadata.read_text().splitlines() if metadata.is_file() else []
        if not lines or lines[0] != revision:
            raise BackendError("Staged model revision is unverified")
        config = json.loads((self.path / "config.json").read_text())
        if config.get("architectures") != ["Qwen3ForCausalLM"]:
            raise BackendError("Only the reviewed native Qwen3 architecture is supported")
        self.device, self.dtype, self.threads = device, dtype, threads
        self.max_input_tokens = max_input_tokens
        self._runtime = None
        self.metadata = {}

    def load(self):
        if self._runtime is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        started = time.monotonic()
        torch.set_num_threads(self.threads)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise BackendError("Allocated GPU is unavailable")
        if self.device == "cuda" and self.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise BackendError("Allocated GPU does not support the requested BF16 runtime")
        tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, self.dtype),
            device_map={"": 0} if self.device == "cuda" else None,
            attn_implementation="sdpa",
        ).eval().requires_grad_(False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        import transformers
        self.metadata = {
            "backend_version": BACKEND_VERSION, "model_id": self.model_id,
            "revision": self.revision, "actual_dtype": str(model.dtype),
            "device": self.device, "threads": self.threads,
            "torch": str(torch.__version__), "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "cuda": torch.version.cuda, "cold_load_seconds": time.monotonic() - started,
            "thinking_enabled": True, "max_input_tokens": self.max_input_tokens,
            "deadline_kind": "per_token_soft_stop_and_posthoc_rejection",
            "shared_state": "weights/tokenizer only; no request dialogue or KV persistence",
        }
        self._runtime = torch, tokenizer, model

    def complete(self, messages, max_new_tokens, temperature, seed, deadline):
        validate_request(messages, max_new_tokens, temperature, seed, deadline)
        started = time.monotonic()
        self.load()
        torch, tokenizer, model = self._runtime
        if time.monotonic() >= deadline:
            raise BackendError("Request expired during cold load")
        from transformers import StoppingCriteria, StoppingCriteriaList

        class DeadlineStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.monotonic() >= deadline

        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            enable_thinking=True,
        )
        input_tokens = inputs["input_ids"].shape[-1]
        if input_tokens > self.max_input_tokens:
            raise BackendError("Input budget exceeded; authoritative facts are never silently truncated")
        inputs = inputs.to(model.device)
        if time.monotonic() >= deadline:
            raise BackendError("Request expired during prefill preparation")
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                  "pad_token_id": tokenizer.eos_token_id,
                  "stopping_criteria": StoppingCriteriaList([DeadlineStop()])}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=.95, top_k=20)
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        if self.device == "cuda":
            torch.cuda.synchronize()
        new_tokens = generated[0, input_tokens:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        eos_ids = model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
        ended_with_eos = bool(len(new_tokens) and int(new_tokens[-1]) in eos_ids)
        finish_reason = ("eos" if ended_with_eos else
                         "length" if len(new_tokens) >= max_new_tokens else "stopping_criteria")
        elapsed = time.monotonic() - started
        if time.monotonic() >= deadline:
            raise BackendError("Late generation rejected; partial reasoning is never an action")
        provenance = dict(self.metadata, seed=seed, temperature=temperature,
                          max_new_tokens=max_new_tokens, input_tokens=input_tokens,
                          output_tokens=len(new_tokens), elapsed_seconds=elapsed,
                          at_token_limit=len(new_tokens) >= max_new_tokens,
                          finish_reason=finish_reason,
                          peak_vram_bytes=(torch.cuda.max_memory_allocated()
                                           if self.device == "cuda" else None))
        # Tensors and generated KV cache are call-local and are not retained.
        return {"text": text, "provenance": provenance}


def serve(socket_path, engine, ready_path=None):
    """Serial service: a queued request's original deadline remains in force."""
    import socketserver
    from http.server import BaseHTTPRequestHandler

    path = Path(socket_path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise BackendError("Socket directory is not owner-only")
    if path.exists():
        raise BackendError("Refuse to replace an existing service socket")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never log prompts, private observations, reasoning or socket callers.

        def do_POST(self):
            status, response = 200, None
            try:
                if self.path != "/complete":
                    raise BackendError("Unsupported endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise BackendError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) != {
                    "messages", "max_new_tokens", "temperature", "seed", "deadline"
                }:
                    raise BackendError("Invalid request keys")
                # Validation occurs after queueing as well as in the client.
                validate_request(**payload)
                response = engine.complete(**payload)
            except (BackendError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                status, response = 422, {"error": type(exc).__name__}
            data = json.dumps(response, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Caller timed out; never retry the generation blindly.

    with socketserver.UnixStreamServer(str(path), Handler) as server:
        try:
            path.chmod(0o600)
            if ready_path is not None:
                # Binding and permission checks precede the readiness receipt.
                ready = Path(ready_path)
                temporary = ready.with_name(ready.name + f".tmp-{os.getpid()}")
                try:
                    with temporary.open("x") as stream:
                        json.dump(engine.metadata, stream, indent=2)
                    # Publish a complete receipt atomically, refusing overwrite.
                    os.link(temporary, ready)
                finally:
                    temporary.unlink(missing_ok=True)
            server.serve_forever(poll_interval=.25)
        finally:
            path.unlink(missing_ok=True)
