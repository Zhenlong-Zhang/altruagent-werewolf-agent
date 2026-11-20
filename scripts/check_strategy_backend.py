"""Slurm-only checks of request privacy, service isolation and queue deadlines."""

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from werewolf_agent.strategy_backend import BackendError, GenerationClient, HFGenerationBackend, serve, validate_request


class EchoEngine:
    metadata = {"backend_version": "test-only-stateless-echo"}

    def __init__(self, entered):
        self.entered = entered

    def complete(self, messages, max_new_tokens, temperature, seed, deadline):
        if messages[-1]["content"] == "slow test request":
            self.entered.set()
            time.sleep(.2)
        return {"text": messages[-1]["content"],
                "provenance": {"backend_version": "test-only-stateless-echo"}}


class FakeBatch(dict):
    def to(self, device):
        return self


class AdapterChecks(unittest.TestCase):
    def fixture(self, input_tokens=4, output=(8, 2), delay=0):
        import torch
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        metadata = root / ".cache/huggingface/download/config.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("pinned-test-revision\n")
        (root / "config.json").write_text('{"architectures":["Qwen3ForCausalLM"]}')
        backend = HFGenerationBackend(root, "synthetic", "pinned-test-revision",
                                      device="cpu", dtype="float32", max_input_tokens=512)

        class Tokenizer:
            eos_token_id = 2

            def apply_chat_template(self, messages, **kwargs):
                self.thinking = kwargs.get("enable_thinking")
                self.truncated = kwargs.get("truncation", False)
                return FakeBatch(input_ids=torch.ones((1, input_tokens), dtype=torch.long),
                                 attention_mask=torch.ones((1, input_tokens), dtype=torch.long))

            def decode(self, tokens, **kwargs):
                self.decoded = tokens.tolist()
                return '<think>private</think><answer>{"action_id":"opaque"}</answer>'

        class Model:
            device = torch.device("cpu")
            generation_config = SimpleNamespace(eos_token_id=[2, 3])
            calls = 0

            def generate(self, input_ids, **kwargs):
                self.calls += 1
                if delay:
                    time.sleep(delay)
                return torch.cat((input_ids, torch.tensor([output], dtype=torch.long)), dim=-1)

        tokenizer, model = Tokenizer(), Model()
        backend._runtime = torch, tokenizer, model
        backend.metadata = {"backend_version": "fake-hf-test-only"}
        return backend, tokenizer, model

    def test_native_reasoning_and_only_new_completion_tokens_are_decoded(self):
        backend, tokenizer, _ = self.fixture()
        answer = backend.complete([{"role": "user", "content": "own view"}], 2, 0, 1,
                                  time.monotonic() + 10)
        self.assertTrue(tokenizer.thinking)
        self.assertEqual(tokenizer.decoded, [8, 2])
        self.assertEqual(answer["provenance"]["input_tokens"], 4)
        self.assertEqual(answer["provenance"]["finish_reason"], "eos")
        self.assertTrue(answer["provenance"]["at_token_limit"])

    def test_token_cap_without_eos_is_reported_as_length_not_successful_eos(self):
        backend, _, _ = self.fixture(output=(8, 9))
        answer = backend.complete([{"role": "user", "content": "own view"}], 2, 0, 1,
                                  time.monotonic() + 10)
        self.assertEqual(answer["provenance"]["finish_reason"], "length")

    def test_oversize_input_is_rejected_without_truncating_protected_facts(self):
        backend, tokenizer, model = self.fixture(input_tokens=513)
        with self.assertRaises(BackendError):
            backend.complete([{"role": "user", "content": "long view"}], 2, 0, 1,
                             time.monotonic() + 10)
        self.assertFalse(tokenizer.truncated)
        self.assertEqual(model.calls, 0)

    def test_cold_load_expiry_prevents_generation(self):
        backend, _, model = self.fixture()
        backend.load = lambda: time.sleep(.04)
        with self.assertRaises(BackendError):
            backend.complete([{"role": "user", "content": "own view"}], 2, 0, 1,
                             time.monotonic() + .01)
        self.assertEqual(model.calls, 0)

    def test_uninterruptible_late_generation_is_rejected_posthoc(self):
        backend, _, _ = self.fixture(delay=.06)
        with self.assertRaises(BackendError):
            backend.complete([{"role": "user", "content": "own view"}], 2, 0, 1,
                             time.monotonic() + .03)


class BackendChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.socket = self.root / "service.sock"
        self.ready = self.root / "ready.json"
        self.entered = multiprocessing.Event()
        self.process = multiprocessing.Process(
            target=serve, args=(self.socket, EchoEngine(self.entered), self.ready))
        self.process.start()
        self.addCleanup(self.shutdown)
        until = time.monotonic() + 5
        while not self.ready.exists() and time.monotonic() < until:
            time.sleep(.01)
        self.assertTrue(self.socket.exists())
        self.assertEqual(json.loads(self.ready.read_text()), EchoEngine.metadata)

    def shutdown(self):
        self.process.terminate()
        self.process.join(3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()

    def test_complete_context_is_not_shared_between_seats(self):
        a, b = GenerationClient(self.socket), GenerationClient(self.socket)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        own_secret = "private-own-seat-A-check-Player4"
        first = a.complete([{"role": "user", "content": own_secret}], 100, 0, 1,
                           time.monotonic() + 3)
        second = b.complete([{"role": "user", "content": "seat-B-public-only"}], 100, 0, 2,
                            time.monotonic() + 3)
        self.assertEqual(first["text"], own_secret)
        self.assertEqual(second["text"], "seat-B-public-only")
        self.assertNotIn(own_secret, json.dumps(second))

    def test_socket_permissions_and_public_directory_rejected(self):
        self.assertEqual(self.socket.stat().st_mode & 0o777, 0o600)
        self.root.chmod(0o755)
        try:
            with self.assertRaises(BackendError):
                GenerationClient(self.socket)
        finally:
            self.root.chmod(0o700)

    def test_expired_and_nonfinite_requests_do_not_reach_the_model(self):
        client = GenerationClient(self.socket)
        self.addCleanup(client.close)
        for deadline in (time.monotonic() - 1, float("nan"), float("inf"), True):
            with self.assertRaises(BackendError):
                client.complete([{"role": "user", "content": "do not generate"}],
                                100, 0, 7, deadline)

    def test_service_revalidates_deadline_after_queueing(self):
        import httpx
        from concurrent.futures import ThreadPoolExecutor

        def slow_request():
            client = GenerationClient(self.socket)
            try:
                return client.complete([{"role": "user", "content": "slow test request"}],
                                       100, 0, 1, time.monotonic() + 3)
            finally:
                client.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(slow_request)
            self.assertTrue(self.entered.wait(2))
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(self.socket))) as client:
                response = client.post("http://strategy.local/complete", json={
                    "messages": [{"role": "user", "content": "expires while queued"}],
                    "max_new_tokens": 100, "temperature": 0, "seed": 1,
                    "deadline": time.monotonic() + .03,
                })
            self.assertEqual(first.result()["text"], "slow test request")
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("text", response.json())

    def test_hidden_session_memory_and_unsupported_request_keys_rejected(self):
        import httpx
        with httpx.Client(transport=httpx.HTTPTransport(uds=str(self.socket))) as client:
            response = client.post("http://strategy.local/complete", json={
                "messages": [{"role": "user", "content": "public"}],
                "max_new_tokens": 100, "temperature": 0, "seed": 1,
                "deadline": time.monotonic() + 3,
                "prior_seat_history": "must not be accepted as a shared session",
            })
        self.assertEqual(response.status_code, 422)

    def test_request_limits_reject_invalid_seed_or_chat_roles(self):
        arguments = ([{"role": "user", "content": "data"}], 100, 0, 1,
                     time.monotonic() + 3)
        for seed in (True, -1, 2**32):
            with self.assertRaises(BackendError):
                validate_request(*arguments[:3], seed, arguments[-1])
        with self.assertRaises(BackendError):
            validate_request([{"role": "tool", "content": "unsupported"}],
                             *arguments[1:])


if __name__ == "__main__":
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run strategy backend checks inside a Slurm allocation")
    unittest.main()
