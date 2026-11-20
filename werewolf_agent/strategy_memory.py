"""Phase-B owner-isolated raw ledger and bounded, immutable actor snapshots."""

from dataclasses import dataclass, field
import copy
import fcntl
import json
import math
from pathlib import Path

from .strategy_state import compile_observation, envelope, fingerprint, seat


SCHEMA_B = "strategy-observation-v2"
MEMORY_VERSION = "source-slots-v1"
CLAIM_CHARS = 256
RECENT_MESSAGES = 6
CLAIMS_PER_PLAYER = 4
FACT_CAPACITY = 32
OWN_ACK_CAPACITY = 64


@dataclass
class Window:
    owner: dict
    accepted: int = 0
    attempted: int = 0
    private_accepted: int = 0
    remaining: int = 5
    stage: str = "opening"
    terminated: bool = False
    active: bool = False
    deadline: float | None = None
    pending: dict | None = None
    consumed_seq: int = -1
    accepted_ids: set = field(default_factory=set)
    accepted_seqs: set = field(default_factory=set)


class StrategyMemory:
    def __init__(self, owner, journal_path):
        if set(owner) != {"session_hash", "seat"} or not isinstance(owner["session_hash"], str):
            raise ValueError("Invalid strategy memory owner")
        seat(owner["seat"])
        self.owner = copy.deepcopy(owner)
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a")
        self.path.with_suffix(self.path.suffix + ".lock").chmod(0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise ValueError("Another writer owns this strategy ledger")
        self.messages, self.windows, self.deliveries, self.assertions = {}, {}, {}, []
        self.current = None
        self.last_state = None
        self.last_signature = None
        self.last_snapshot_at = -1.
        self.last_flush_at = -1.
        self.flush_error = None
        self.role = None
        self.facts = {}
        if self.path.exists():
            self._replay()

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def _check_context(self, context):
        import hashlib
        expected = {"session_hash": hashlib.sha256(context.session_id.encode()).hexdigest()[:16],
                    "seat": seat(context.seat_position)}
        if expected != self.owner:
            raise ValueError("Cross-owner strategy state access")

    @staticmethod
    def _time(value):
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Invalid monotonic observation time")
        return float(value)

    def record(self, event, **data):
        row = copy.deepcopy({"event": event, "owner": self.owner, **data})
        with self.path.open("a", encoding="utf-8") as stream:
            self.path.chmod(0o600)
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return row

    def _replay(self):
        # Replay persisted state checkpoints only; never infer acceptance from
        # a prepared request. Monotonic deadlines do not survive process restart.
        for line in self.path.read_text().splitlines():
            row = json.loads(line)
            if row.get("owner") != self.owner:
                raise ValueError("Cross-owner journal record")
            if row.get("event") in {"visible_message", "visible_annotation"}:
                message = envelope(row["message"], self.owner["seat"])
                self.messages[message["seq"]] = message
            elif row.get("event") == "memory_checkpoint":
                self.deliveries = row["deliveries"]
                self.assertions = row["assertions"]
                self.facts = row.get("facts", {})
                self.role = row.get("role")
                for key, data in row["windows"].items():
                    value = dict(data)
                    value["accepted_ids"] = set(value["accepted_ids"])
                    value["accepted_seqs"] = set(value["accepted_seqs"])
                    value["deadline"], value["active"] = None, False
                    if value["pending"] and value["pending"]["status"] == "prepared":
                        value["pending"]["status"] = "unknown"
                    self.windows[int(key)] = Window(**value)
                    if value["pending"]:
                        self._validate_pending(value["pending"])
        for window in self.windows.values():
            if window.pending:
                identifier = window.pending["delivery_id"]
                if identifier not in self.deliveries:
                    raise ValueError("Replayed pending has no request record")
                window.pending = self.deliveries[identifier]
                if window.pending["status"] == "prepared":
                    window.pending["status"] = "unknown"

    def _validate_pending(self, value):
        keys = {"owner", "delivery_id", "decision_id", "candidate_id", "intent", "target", "recipients",
                "canonical_text", "assertions", "assertion_realizations", "packet_hash", "send_floor_seq",
                "attempted_at", "status"}
        if not isinstance(value, dict) or set(value) - {"accepted_seq"} != keys:
            raise ValueError("Invalid typed pending fields")
        owner = value["owner"]
        if (not isinstance(owner, dict) or set(owner) != {"session_hash", "seat", "day", "window_kind"}
                or any(owner[key] != self.owner[key] for key in self.owner)
                or type(owner["day"]) is not int or owner["day"] < 0
                or owner["window_kind"] != "day_vote_discussion"):
            raise ValueError("Pending record belongs to a different owner/window")
        for key in ("delivery_id", "candidate_id", "intent", "canonical_text", "packet_hash", "status"):
            if not isinstance(value[key], str) or not value[key]:
                raise ValueError("Pending scalar must be a nonempty string")
        if value["status"] not in {"prepared", "unknown", "accepted", "rejected"}:
            raise ValueError("Invalid pending status")
        if value["intent"] not in {"disclose", "ask_evidence", "challenge", "defend", "coordinate_vote", "withhold", "bluff", "coordinate_private", "safe_fallback"}:
            raise ValueError("Invalid pending intent")
        if len(value["canonical_text"]) > 1024 or len(value["canonical_text"].split()) > 50:
            raise ValueError("Invalid pending text limit")
        if type(value["decision_id"]) is not int or value["decision_id"] < 0 or type(value["send_floor_seq"]) is not int:
            raise ValueError("Invalid pending decision/sequence")
        if "accepted_seq" in value and (type(value["accepted_seq"]) is not int or value["accepted_seq"] <= value["send_floor_seq"]):
            raise ValueError("Invalid pending accepted sequence")
        self._time(value["attempted_at"])
        if value["target"] is not None:
            seat(value["target"])
        if not isinstance(value["recipients"], list) or len(value["recipients"]) > 1:
            raise ValueError("Invalid pending recipients")
        for recipient in value["recipients"]:
            seat(recipient)
        if (not isinstance(value["assertions"], list) or len(value["assertions"]) > 2
                or not isinstance(value["assertion_realizations"], list)):
            raise ValueError("Invalid pending assertion records")
        for assertion in value["assertions"]:
            if not isinstance(assertion, dict) or set(assertion) != {"subject", "predicate", "value", "source_ref", "truth_mode"}:
                raise ValueError("Invalid nested pending assertion")
            seat(assertion["subject"])
            if (not all(isinstance(assertion[key], str) for key in ("predicate", "value", "truth_mode"))
                    or assertion["source_ref"] is not None and not isinstance(assertion["source_ref"], str)):
                raise ValueError("Invalid nested pending assertion scalar")
            if (assertion["predicate"] not in {"alignment", "role", "investigation"}
                    or assertion["value"] not in ({"wolf", "seer", "villager"} if assertion["predicate"] == "role" else {"wolf", "nonwolf"})
                    or assertion["truth_mode"] not in {"verified", "attributed", "uncertain", "deliberate_bluff"}):
                raise ValueError("Invalid nested pending assertion enum")
        if len(value["assertion_realizations"]) != len(value["assertions"]):
            raise ValueError("Pending assertion realization length mismatch")
        if any(not isinstance(mode, str) or mode not in {"verified", "uncertain", "deliberate_bluff", "source_message_clarification"}
               for mode in value["assertion_realizations"]):
            raise ValueError("Invalid pending realization marker")

    def _checkpoint(self):
        windows = {}
        for day, value in self.windows.items():
            windows[str(day)] = {**vars(value), "accepted_ids": sorted(value.accepted_ids),
                                 "accepted_seqs": sorted(value.accepted_seqs)}
        self.record("memory_checkpoint", windows=windows, deliveries=self.deliveries, assertions=self.assertions,
                    facts=self.facts, role=self.role)

    def observe_state(self, state, context, observed_at):
        self._check_context(context)
        now = self._time(observed_at)
        signature = fingerprint({"phase": state.phase, "game": state.raw.get("game_state"),
                                 "seconds": state.raw.get("messaging_seconds_left"),
                                 "terminated": getattr(state, "terminated_messaging", []),
                                 "messages": [envelope(message, self.owner["seat"]) for message in state.new_messages]})
        if self.last_state is state and self.last_signature == signature:
            return
        self.last_state = state
        self.last_signature = signature
        self.last_snapshot_at = now
        game = state.raw.get("game_state") or {}
        day = game.get("day")
        if type(day) is not int or day < 0:
            raise ValueError("State has no valid day/window owner")
        owner = {**self.owner, "day": day, "window_kind": "day_vote_discussion"}
        window = self.windows.setdefault(day, Window(owner))
        if window.owner != owner:
            raise ValueError("Cross-window state collision")
        self.current = window
        window.active = state.phase == "messaging"
        seconds = state.raw.get("messaging_seconds_left")
        if seconds is not None:
            seconds = self._time(seconds)
            if seconds < 0:
                raise ValueError("Negative server phase time")
            deadline = now + seconds
            window.deadline = min(window.deadline, deadline) if window.deadline is not None else deadline
        if self.owner["seat"] in getattr(state, "terminated_messaging", []):
            window.terminated, window.stage = True, "terminated"
        stamped = []
        for message in state.new_messages:
            value = envelope(message, self.owner["seat"])
            value["day"] = day
            stamped.append(value)
        self.merge_visible(stamped, "state_snapshot", False, now)

    def merge_visible(self, envelopes, source, complete, observed_at):
        now = self._time(observed_at)
        if not isinstance(envelopes, list) or type(complete) is not bool:
            raise ValueError("Malformed authorized transcript update")
        for message in envelopes:
            value = envelope(message, self.owner["seat"])
            old = self.messages.get(value["seq"])
            if old is not None and old != value:
                # Server transcript may omit day metadata present in a local
                # snapshot. Do not treat that optional annotation as new text.
                if any(old[key] != value[key] for key in ("seq", "sender", "recipients", "channel", "type", "content")):
                    raise ValueError("Conflicting visible message sequence")
                for key in ("day", "move_index"):
                    value[key] = old[key] if old[key] is not None else value[key]
            if old is None:
                self.messages[value["seq"]] = value
                self.record("visible_message", source=source, message=value)
            elif old != value:
                self.messages[value["seq"]] = value
                self.record("visible_annotation", source=source, message=value)
        if complete:
            self.last_flush_at, self.flush_error = now, None
        self.reconcile()

    def mark_flush_error(self, error):
        self.flush_error = str(error)
        self.record("transcript_refresh_error", error_type=str(error))

    def remaining_seconds(self, now):
        now = self._time(now)
        return max(0., self.current.deadline - now) if self.current and self.current.deadline is not None else 0.

    def can_chat(self, now, reserve=3.):
        value = self.current
        return bool(value and value.active and not value.terminated and value.accepted < 3
                    and value.attempted < 4 and value.remaining > 0 and value.pending is None
                    and self.remaining_seconds(now) > reserve)

    def relevant_update(self):
        if not self.current:
            return False
        seen = {(row["sender"], row["content"], tuple(row["recipients"])) for row in self.messages.values()
                if row["seq"] <= self.current.consumed_seq}
        subjects = {self.owner["seat"], *(fact["subject"] for fact in self.facts.values())}
        if self.last_state is not None:
            from .strategy_state import legal_record
            phase = (self.last_state.raw.get("game_state") or {}).get("phase")
            subjects.update(record["target"] for record in (legal_record(action, phase) for action in self.last_state.legal_actions)
                            if record["target"] is not None)
        commitments = [row["target"] for row in self.assertions if row["intent"] == "coordinate_vote"]
        subjects.update(target for target in commitments if target is not None)
        for row in sorted(self.messages.values(), key=lambda item: item["seq"]):
            key = (row["sender"], row["content"], tuple(row["recipients"]))
            if (row["seq"] > self.current.consumed_seq and row["sender"] != self.owner["seat"]
                    and row["type"] == "chat" and key not in seen):
                import re
                text = row["content"].lower()
                mentions = {int(value) for value in re.findall(r"\bplayer\s*([0-6])\b", text)}
                if row["recipients"] or "seer" in text or mentions & subjects:
                    return True
            seen.add(key)
        return False

    def prepare(self, delivery_id, decision_id, candidate, packet, now):
        self._time(now)
        if not isinstance(delivery_id, str) or not delivery_id or delivery_id in self.deliveries:
            raise ValueError("Duplicate/invalid local delivery ID")
        if not self.can_chat(now):
            raise ValueError("No legal strategy chat budget or unresolved delivery")
        if packet["owner"] != self.owner or packet["window"]["owner"] != self.current.owner:
            raise ValueError("Prepared request belongs to another owner/window")
        if candidate.get("delivery") != "chat" or candidate.get("realization_mode") != "typed-canonical-v1":
            raise ValueError("Only validated canonical chat may be prepared")
        from .strategy_candidates_v2 import validate_delivery_candidate
        validate_delivery_candidate(candidate, packet)
        recipients = candidate["recipients"]
        if recipients and self.current.private_accepted:
            raise ValueError("Private coordination already accepted in this window")
        if not isinstance(candidate["text"], str) or not candidate["text"] or len(candidate["text"]) > 1024 or len(candidate["text"].split()) > 50:
            raise ValueError("Invalid canonical delivery content")
        value = {"owner": copy.deepcopy(self.current.owner), "delivery_id": delivery_id,
                 "decision_id": decision_id, "candidate_id": candidate["id"], "intent": candidate["intent"],
                 "target": candidate["target"], "recipients": copy.deepcopy(recipients),
                 "canonical_text": candidate["text"], "assertions": copy.deepcopy(candidate["assertions"]),
                 "assertion_realizations": copy.deepcopy(candidate.get("assertion_realizations", [])),
                 "packet_hash": packet["packet_hash"], "send_floor_seq": max(self.messages, default=-1),
                 "attempted_at": now, "status": "prepared"}
        self._validate_pending(value)
        self.current.attempted += 1
        self.current.pending = value
        self.deliveries[delivery_id] = value
        self.current.consumed_seq = max(self.messages, default=-1)
        self.record("delivery_prepared", pending=value)
        self._checkpoint()

    def _accept(self, pending, sequence=None):
        day = pending["owner"]["day"]
        window = self.windows[day]
        identifier = pending["delivery_id"]
        if identifier not in window.accepted_ids:
            window.accepted_ids.add(identifier)
            window.accepted += 1
            window.remaining = max(0, window.remaining - 1)
            window.private_accepted += bool(pending["recipients"])
            window.stage = "response" if window.accepted == 1 else "commitment"
            self.assertions.append({"delivery_id": identifier, "owner": pending["owner"],
                                    "intent": pending["intent"], "target": pending["target"],
                                    "recipients": pending["recipients"], "assertions": pending["assertions"],
                                    "assertion_realizations": pending["assertion_realizations"],
                                    "accepted_seq": sequence, "original_packet_hash": pending["packet_hash"]})
        if sequence is not None:
            window.accepted_seqs.add(sequence)
            pending["accepted_seq"] = sequence
            for assertion in self.assertions:
                if assertion["delivery_id"] == identifier:
                    assertion["accepted_seq"] = sequence
        pending["status"] = "accepted"
        if window.pending and window.pending["delivery_id"] == identifier:
            window.pending = None

    def apply_delivery(self, delivery_id, result, context, observed_at):
        self._check_context(context)
        self._time(observed_at)
        pending = self.deliveries.get(delivery_id)
        if pending is None:
            raise ValueError("Result has no owned prepared request")
        self._validate_pending(pending)
        if not isinstance(result, dict):
            raise ValueError("Delivery result must be a transport object")
        window = self.windows[pending["owner"]["day"]]
        if result.get("accepted") is True and not result.get("error"):
            if pending["status"] == "rejected":
                raise ValueError("Conflicting rejected/accepted delivery receipts")
            self._accept(pending)
        elif result.get("accepted") is False or result.get("error") in {
                "INVALID_CONTENT", "INVALID_RECIPIENTS", "TOO_MANY_WORDS", "MESSAGE_TOO_LONG",
                "MESSAGES_QUOTA_EXCEEDED", "WRONG_PHASE", "PLAYER_ELIMINATED"}:
            if pending["status"] == "accepted":
                raise ValueError("Conflicting accepted/rejected delivery receipts")
            pending["status"] = "rejected"
            if window.pending and window.pending["delivery_id"] == delivery_id:
                window.pending = None
        elif pending["status"] != "accepted":
            pending["status"] = "unknown"
        remaining = (result.get("remaining") or {}).get("this_turn")
        if remaining is not None:
            if type(remaining) is not int or not 0 <= remaining <= 5:
                raise ValueError("Invalid authoritative quota response")
            window.remaining = min(window.remaining, remaining)
        self.record("delivery_result", delivery_id=delivery_id, status=pending["status"],
                    accepted=result.get("accepted"), error=result.get("error"))
        self._checkpoint()

    def reconcile(self):
        assigned = {value.get("accepted_seq") for value in self.deliveries.values() if value.get("accepted_seq") is not None}
        possible = {}
        for identifier, pending in self.deliveries.items():
            if pending.get("accepted_seq") is not None or pending["status"] == "rejected":
                continue
            possible[identifier] = [row for row in self.messages.values()
                if row["type"] == "chat" and row["sender"] == self.owner["seat"]
                and row["seq"] > pending["send_floor_seq"] and row["content"] == pending["canonical_text"]
                and row["recipients"] == pending["recipients"] and row["seq"] not in assigned
                and (row["day"] == pending["owner"]["day"] or row["day"] is None
                     and self.current is not None and self.current.owner == pending["owner"])]
        consumers = {}
        for identifier, matches in possible.items():
            for message in matches:
                consumers.setdefault(message["seq"], set()).add(identifier)
        changed = False
        for identifier, matches in possible.items():
            pending = self.deliveries[identifier]
            if len(matches) == 1 and len(consumers[matches[0]["seq"]]) == 1:
                self._accept(pending, matches[0]["seq"])
                assigned.add(matches[0]["seq"])
                self.record("delivery_reconciled", delivery_id=pending["delivery_id"], sequence=matches[0]["seq"])
                changed = True
        if changed:
            self._checkpoint()

    def terminate(self, accepted=False):
        if self.current:
            self.current.stage = "terminated"
            self.current.terminated = bool(accepted) or self.current.terminated
            self._checkpoint()

    def semantic_assertions(self):
        """Keep first/latest ACK provenance per semantic assertion or vote intent.

        Exact raw message records remain in the journal; duplicate rhetoric is
        not a new protected assertion. The learner uses the same semantic key.
        """
        semantic = {}
        for record in self.assertions:
            entries = list(enumerate(record["assertions"]))
            if record["intent"] == "coordinate_vote":
                entries.append((None, {"subject": record["target"], "predicate": "vote_commitment", "value": "none",
                                       "source_ref": None, "truth_mode": "uncertain"}))
            for index, assertion in entries:
                key = fingerprint({"assertion": assertion, "recipients": record["recipients"], "intent": record["intent"]})
                value = copy.deepcopy(record)
                value["assertions"] = [] if index is None else [assertion]
                value["assertion_realizations"] = [] if index is None else [record["assertion_realizations"][index]]
                semantic.setdefault(key, [value, value])[1] = value
        if len(semantic) > OWN_ACK_CAPACITY:
            raise ValueError("Protected semantic own-assertion capacity exceeded")
        output = {}
        for pair in semantic.values():
            for record in pair:
                output[fingerprint(record)] = record
        chronology = {record["delivery_id"]: index for index, record in enumerate(self.assertions)}
        return sorted(output.values(), key=lambda record: chronology[record["delivery_id"]])

    def compile_packet(self, state, context, token_counter, input_budget=8192, now=None):
        self._check_context(context)
        if not callable(token_counter) or type(input_budget) is not int or input_budget < 1:
            raise ValueError("An exact complete-prompt token counter is required")
        from types import SimpleNamespace
        snapshot = SimpleNamespace(**{**vars(state), "new_messages": []})
        base = compile_observation(snapshot, context, history={"owner": self.owner,
                                  "messages": [self.messages[index] for index in sorted(self.messages)]})
        if self.role is not None and self.role != base["identity"]["role"]:
            raise ValueError("Own role changed within fixed-role session")
        self.role = base["identity"]["role"]
        for fact in base["verified_facts"]:
            for previous in self.facts.values():
                if previous["subject"] == fact["subject"] and (previous["alignment"] != fact["alignment"]
                        or previous["role"] and fact["role"] and previous["role"] != fact["role"]):
                    raise ValueError("Persistent authoritative fact contradiction")
            if fact["id"] in self.facts and self.facts[fact["id"]] != fact:
                raise ValueError("Persistent fact source changed")
            self.facts[fact["id"]] = fact
        base["verified_facts"] = [copy.deepcopy(self.facts[key]) for key in sorted(self.facts)]
        for player in base["player_records"]:
            known = [fact for fact in base["verified_facts"] if fact["subject"] == player["seat"]]
            player["verified_refs"] = [fact["id"] for fact in known]
            player["alignment"] = known[0]["alignment"] if known else "unknown"
            if known:
                base["belief_scores"][str(player["seat"])] = 4. if known[0]["alignment"] == "wolf" else -4.
        if len(base["verified_facts"]) > FACT_CAPACITY:
            raise ValueError("Protected fact capacity exceeded")
        own = self.semantic_assertions()
        claims = base["attributed_claims"]
        selected = []
        fact_subjects = {fact["subject"] for fact in base["verified_facts"]}
        for player in range(7):
            records = [claim for claim in claims if claim["subject_speaker"] == player]
            chosen = records[-2:]
            older = records[:-2]
            relevant = [claim for claim in older if any(f"Player{target}" in claim["text"] for target in fact_subjects)]
            if relevant:
                chosen.append(relevant[-1])
            for claim in reversed(older):
                if claim not in chosen:
                    chosen.append(claim)
                    break
            selected.extend(chosen[:CLAIMS_PER_PLAYER])
        selected = list({claim["id"]: claim for claim in selected}.values())
        projection = copy.deepcopy(base)
        projection["schema_version"] = SCHEMA_B
        projection["self_assertions"] = copy.deepcopy(own)
        projection["prior_ballots"] = projection["prior_ballots"][-3:]
        if self.current.pending:
            self._validate_pending(self.current.pending)
        projection["window"] = {
            "owner": copy.deepcopy(self.current.owner), "accepted": self.current.accepted,
            "attempted": self.current.attempted, "remaining": self.current.remaining,
            "private_accepted": self.current.private_accepted, "stage": self.current.stage,
            "terminated": self.current.terminated, "pending": copy.deepcopy(self.current.pending),
            "transcript_complete": self.last_flush_at >= self.last_snapshot_at and self.flush_error is None,
            "transcript_error": self.flush_error, "seconds_left": self.remaining_seconds(self.last_snapshot_at if now is None else now),
            "highest_visible_seq": max(self.messages, default=-1),
        }
        projection["visible_dialogue"] = []
        projection["attributed_claims"] = []
        for player in projection["player_records"]:
            player["claim_refs"] = []
        projection["provenance"].update(memory=MEMORY_VERSION, dropped_claim_count=len(claims),
                                          excerpt_char_cap=CLAIM_CHARS, full_ledger_messages=len(self.messages),
                                          ballot_projection="latest-three-v1", dropped_ballot_count=max(0, len(base["prior_ballots"]) - 3),
                                          protected_own_ack_capacity=OWN_ACK_CAPACITY)

        def finalize():
            projection.pop("packet_hash", None)
            projection["packet_hash"] = fingerprint(projection)
            count = token_counter(projection)
            if type(count) is not int or count < 1:
                raise ValueError("Invalid native prompt token count")
            return count

        if finalize() > input_budget:
            raise ValueError("Protected observation exceeds native input budget")
        for claim in sorted(selected, key=lambda row: int(row["id"].split(":")[1]), reverse=True):
            excerpt = copy.deepcopy(claim)
            excerpt["full_source_hash"] = fingerprint(claim)
            excerpt["excerpt_truncated"] = len(excerpt["text"]) > CLAIM_CHARS
            excerpt["text"] = excerpt["text"][:CLAIM_CHARS]
            projection["attributed_claims"].append(excerpt)
            if finalize() > input_budget:
                projection["attributed_claims"].pop()
        retained = {claim["message_ref"] for claim in projection["attributed_claims"]}
        for message in [self.messages[index] for index in sorted(self.messages, reverse=True)]:
            if message["id"] in retained or len(projection["visible_dialogue"]) >= RECENT_MESSAGES:
                continue
            excerpt = copy.deepcopy(message)
            excerpt["full_source_hash"] = fingerprint(message)
            excerpt["excerpt_truncated"] = len(excerpt["content"]) > CLAIM_CHARS
            excerpt["content"] = excerpt["content"][:CLAIM_CHARS]
            projection["visible_dialogue"].append(excerpt)
            if finalize() > input_budget:
                projection["visible_dialogue"].pop()
        for player in projection["player_records"]:
            player["claim_refs"] = [claim["id"] for claim in projection["attributed_claims"]
                                     if claim["subject_speaker"] == player["seat"]]
        projection["provenance"]["dropped_claim_count"] = len(claims) - len(projection["attributed_claims"])
        if finalize() > input_budget:
            raise ValueError("Final observation metadata exceeds native input budget")
        return projection
