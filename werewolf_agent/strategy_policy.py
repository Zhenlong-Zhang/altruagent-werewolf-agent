"""Opt-in Phase-B teacher/learned strategy policy over owned full-state packets."""

import copy
import hashlib
import math
import json
from pathlib import Path
import time

from altruagent import SendMessage, TERMINATE_MESSAGING
from .strategy_candidates import parse_strategy_answer
from .strategy_candidates_v2 import build_candidate_request, teacher_candidates, validate_delivery_candidate
from .strategy_dialogue import DialogueController
from .strategy_memory import StrategyMemory
from .strategy_state import generation_seed


class StrategyPolicy:
    requires_visible_transcript_flush = True

    def __init__(self, context, backend, token_counter, journal_path, selector=None,
                 model_seconds=30., max_new_tokens=2048, temperature=.6, seed=97001, now=time.monotonic, input_budget=8192):
        owner = {"session_hash": hashlib.sha256(context.session_id.encode()).hexdigest()[:16], "seat": context.seat_position}
        self.memory = StrategyMemory(owner, Path(journal_path))
        self.controller = DialogueController(self.memory)
        self.backend, self.token_counter, self.selector = backend, token_counter, selector
        self.model_seconds, self.max_new_tokens, self.temperature, self.seed = model_seconds, max_new_tokens, temperature, seed
        self.input_budget = input_budget
        if type(input_budget) is not int or not 1 <= input_budget <= 8192:
            raise ValueError("Invalid native input budget")
        if not 0 < model_seconds <= 30 or not callable(token_counter):
            raise ValueError("Strategy requires an exact token counter and a bounded model budget")
        self.now = now
        self.number = max((row.get("decision_id", 0) for row in
                           (json.loads(line) for line in self.memory.path.read_text().splitlines())
                           if row.get("event") == "strategy_request"), default=0) if self.memory.path.exists() else 0
        self.prepared, self.last_action = None, None

    def on_visible_update(self, state, context, envelopes, source, complete, observed_at):
        self.memory.observe_state(state, context, observed_at)
        if source == "refresh_error":
            self.memory.mark_flush_error("authorized_refresh_failed")
        else:
            self.memory.merge_visible(envelopes, source, complete, observed_at)

    def _packet(self, state, context, task):
        self.memory.observe_state(state, context, self.now())
        return self.memory.compile_packet(state, context,
            lambda packet: self.token_counter(build_candidate_request(packet, task)), input_budget=self.input_budget, now=self.now())

    def _fallback(self, packet, task):
        if task == "speech":
            return [{"id": "speech:pass", "intent": "pass", "target": None, "recipients": [],
                     "assertions": [], "assertion_realizations": [], "text": "", "delivery": "terminate",
                     "realization_mode": "typed-canonical-v1", "packet_hash": packet["packet_hash"],
                     "schema_version": "strategy-candidates-v2", "available": True, "unavailable_reason": None}]
        records = [dict(item, intent=item["kind"], delivery="action", packet_hash=packet["packet_hash"],
                        schema_version="strategy-candidates-v2", available=True, unavailable_reason=None)
                   for item in packet["legal_actions"]]
        if not records:
            raise ValueError("Engine offered no legal action")
        return records

    def _decide(self, state, context, task):
        packet = self._packet(state, context, task)
        native_input_tokens = self.token_counter(build_candidate_request(packet, task))
        self.number += 1
        started = self.now()
        deadline = started + self.model_seconds
        if task == "speech":
            deadline = min(deadline, started + self.memory.remaining_seconds(started) - self.controller.reserve)
        teacher, error, provenance = None, None, {}
        try:
            if deadline <= started:
                raise TimeoutError("No model budget before phase delivery reserve")
            reply = self.backend.complete(build_candidate_request(packet, task), self.max_new_tokens,
                                          self.temperature, generation_seed(self.seed, self.memory.owner, self.number), deadline)
            provenance = reply["provenance"]
            answer = parse_strategy_answer(reply["text"], finish_reason=provenance.get("finish_reason",
                "length" if provenance.get("at_token_limit") else "stop"))
            candidates, teacher = teacher_candidates(answer, packet, task)
            if self.now() > deadline:
                raise TimeoutError("Late strategy answer")
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            error = type(exc).__name__
            candidates = self._fallback(packet, task)
        if task == "speech":
            candidates = self.controller.candidates(candidates, self.now())
        else:
            candidates = [dict(candidate, available=True, unavailable_reason=None) for candidate in candidates]
        preference = teacher["index"] if teacher else 0
        requested = preference
        selection = None
        if error is None and self.selector is not None:
            try:
                proposed = self.selector.choose(packet, candidates, deadline=deadline, decision_id=self.number)
                selection = None
                requested = proposed["index"]
                probabilities = proposed["probabilities"]
                if (type(requested) is not int or not 0 <= requested < len(candidates)
                        or len(probabilities) != len(candidates)
                        or any(not isinstance(p, (float, int)) or not math.isfinite(p) or p < 0 for p in probabilities)
                        or not math.isclose(sum(probabilities), 1., abs_tol=1e-9)
                        or any(p and not candidate["available"] for p, candidate in zip(probabilities, candidates))
                        or probabilities[requested] <= 0 or not math.isfinite(proposed["value"])
                        or not math.isclose(proposed["log_probability"], math.log(probabilities[requested]), abs_tol=1e-8)):
                    raise ValueError("Invalid exact contextual policy request")
                json.dumps(proposed, allow_nan=False)
                selection = proposed
                if self.now() > deadline:
                    raise TimeoutError("Late contextual policy request")
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                error = "selector_" + type(exc).__name__
        requested_candidate = candidates[requested] if type(requested) is int and 0 <= requested < len(candidates) else None
        gate = requested_candidate is None or not requested_candidate["available"] or error is not None
        if gate:
            if task == "speech":
                selected = next(index for index, candidate in enumerate(candidates) if candidate["delivery"] == "terminate")
            else:
                role = packet["identity"]["role"]
                scores = packet["belief_scores"]
                selected = max(range(len(candidates)), key=lambda index: (
                    -scores.get(str(candidates[index]["target"]), 0.) if role == "wolf"
                    else scores.get(str(candidates[index]["target"]), 0.)))
        else:
            selected = requested
        record = self.memory.record("strategy_request", schema_version="strategy-transition-v1",
            decision_id=self.number, task=task, packet=packet, candidates=candidates,
            requested_index=requested, teacher_choice=teacher, teacher_source="actual_generation" if teacher else "explicit_fallback",
            behavior_selection=selection, behavior_kind=(
                "greedy" if selection and selection.get("metadata", {}).get("mode") == "greedy" else
                "forced_singleton" if selection and sum(candidate["available"] for candidate in candidates) == 1 else
                "categorical" if selection else "teacher_demo" if teacher and error is None else "forced_fallback"),
            bc_eligible=bool(teacher and selection is None and error is None and not gate),
            executor_index=selected, executor_overridden=gate, error_type=error,
            generator_provenance=provenance, generator_schema="strategy-candidates-v2",
            native_input_tokens=native_input_tokens,
            generator_configuration={"schema": "strategy-candidates-v2", "realization": "typed-canonical-v1",
                "memory": "source-slots-v1", "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature, "model_seconds": self.model_seconds,
                "input_budget": self.input_budget, "accepted_chat_cap": 3, "attempt_cap": 4, "private_chat_cap": 1},
            before_packet_hash=packet["packet_hash"], elapsed_seconds=self.now() - started)
        return packet, candidates[selected], record

    def choose_message(self, state, context):
        self.memory.observe_state(state, context, self.now())
        permitted, reason = self.controller.should_generate(self.now())
        if not permitted:
            self.memory.record("forced_terminate", reason=reason, window_owner=self.memory.current.owner)
            return TERMINATE_MESSAGING
        try:
            packet, candidate, request = self._decide(state, context, "speech")
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.prepared = None
            self.memory.record("unsupported_observation_fallback", task="speech", error_type=type(exc).__name__)
            return TERMINATE_MESSAGING
        self.prepared = (packet, candidate, request)
        if candidate["delivery"] == "terminate":
            return TERMINATE_MESSAGING
        validate_delivery_candidate(candidate, packet)
        return SendMessage(candidate["text"], recipients=candidate["recipients"])

    def choose_action(self, state, context):
        self.memory._check_context(context)
        try:
            packet, candidate, request = self._decide(state, context, "action")
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            self.last_action = None
            self.memory.record("unsupported_observation_fallback", task="action", error_type=type(exc).__name__)
            if not state.legal_actions:
                raise ValueError("No engine legal action for unsupported observation") from exc
            return state.legal_actions[0]
        self.last_action = request
        return next(action for action in state.legal_actions if action.action_id == candidate["id"])

    def before_message_request(self, delivery_id, decision, context, observed_at):
        self.memory._check_context(context)
        if decision is TERMINATE_MESSAGING:
            self.memory.record("terminate_requested", delivery_id=delivery_id)
            return
        if self.prepared is None:
            raise ValueError("Message has no owned policy request")
        packet, candidate, request = self.prepared
        permitted, reason = self.controller.admissible(candidate, observed_at)
        if not permitted or decision.content != candidate["text"] or decision.recipients != candidate["recipients"]:
            self.memory.record("executor_presend_rejected", decision_id=request["decision_id"], reason=reason)
            raise ValueError("Latest known phase/time/controller state rejected the request")
        self.memory.prepare(delivery_id, request["decision_id"], candidate, packet, observed_at)

    def on_delivery_result(self, delivery_id, result, context, observed_at):
        if delivery_id in self.memory.deliveries:
            self.memory.apply_delivery(delivery_id, result, context, observed_at)
        else:
            self.memory._check_context(context)
            self.memory.terminate(accepted=result.get("accepted") is True and not result.get("error"))
            self.memory.record("terminate_result", delivery_id=delivery_id, accepted=result.get("accepted"), error=result.get("error"))

    def on_action_result(self, result, context):
        self.memory._check_context(context)
        self.memory.record("action_result", decision_id=self.last_action["decision_id"] if self.last_action else None,
                           accepted=result.get("accepted"), error=result.get("error"))

    def on_game_result(self, state, context):
        self.memory._check_context(context)
        final = getattr(state, "final_result", None)
        official = isinstance(final, dict) and bool(final)
        score = final.get("your_return") if official else None
        source = "final_result.your_return" if score is not None else None
        returns = getattr(state, "returns", None)
        if score is None and isinstance(returns, dict) and context.agent_id in returns:
            score, source = returns[context.agent_id], "state.returns[context.agent_id]"
        session = getattr(state, "session_id", None)
        consistent = session == context.session_id and (not official or final.get("session_id", session) == session)
        finite_score = type(score) in (int, float) and math.isfinite(score) and score in {-1, 1}
        audit = {"official_final_result_present": official, "is_terminal": getattr(state, "is_terminal", False) is True,
                 "status": getattr(state, "status", None), "session_consistent": consistent,
                 "state_version": getattr(state, "state_version", None), "own_return_source": source,
                 "own_score": score if type(score) in (int, float) and math.isfinite(score) else None,
                 "verified": bool(official and getattr(state, "is_terminal", False) is True and consistent and finite_score)}
        self.memory.record("strategy_terminal", score=audit["own_score"], termination_reason=state.termination_reason,
                           terminal_audit=audit)
        self.memory.close()
