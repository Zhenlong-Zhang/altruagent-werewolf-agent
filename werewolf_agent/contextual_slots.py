"""Deterministic, seat-visible structured slots; no model execution here."""

import copy
import math
import re

from .strategy_candidates_v2 import check_packet, validate_delivery_candidate
from .strategy_state import fingerprint, seat


SERIALIZER = "contextual-slots-v1"
ROLES = ("wolf", "seer", "villager", "unknown")
ALIGNMENTS = ("unknown", "wolf", "nonwolf")
SOURCES = ("own_role", "own_check", "known_partner", "public_engine")
INTENTS = ("disclose", "ask_evidence", "challenge", "defend", "coordinate_vote",
           "withhold", "bluff", "coordinate_private", "none")
MODES = ("verified", "attributed", "uncertain", "deliberate_bluff")
PREDICATES = ("alignment", "role", "investigation", "vote_commitment")
VALUES = ("wolf", "nonwolf", "seer", "villager", "none")
KINDS = ("day_vote", "night_wolf", "night_seer", "abstain", "public_chat", "private_chat", "pass")
PLAYER = re.compile(r"\bPlayer\s*([0-6])\b", re.I)
FORBIDDEN = {"winner", "reward", "terminal_reward", "ground_truth", "all_roles", "true_roles",
             "evaluation", "evaluation_roles", "terminal_result", "outcome", "returns", "advantages"}
DIMENSIONS = (27, 17, 13, 12, 29, 32, 32)
RELATIONS = ("subject", "speaker", "target", "recipient", "same_subject", "source_reference",
             *(f"lexical_entity_{index}" for index in range(7)),
             *(f"public_vote_recency_{index}" for index in range(3)))


def one(value, options):
    if value not in options:
        raise ValueError(f"Unsupported typed category {value!r}")
    return [float(value == item) for item in options]


def bounded(value, cap):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Nonfinite/untyped numeric observation")
    return min(max(float(value), 0.), cap) / cap


def scan(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ValueError("Future/evaluation metadata cannot enter actor slots")
        for child in value.values():
            scan(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            scan(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite packet field")


def source_sequence(row):
    value = row.get("seq", row.get("accepted_seq"))
    if value is None and "message_ref" in row:
        value = int(row["message_ref"].split(":")[-1])
    return value if type(value) is int else -1


def normalized_text(text):
    if not isinstance(text, str) or len(text) > 4096:
        raise ValueError("Slot text exceeds its declared bounded source")
    entities = list(dict.fromkeys(int(value) for value in PLAYER.findall(text)))
    return (PLAYER.sub(lambda match: f"ENTITY_{entities.index(int(match[1]))}", text)
            or "empty typed slot"), entities


def assertion_types(assertions):
    if not isinstance(assertions, list) or len(assertions) > 2:
        raise ValueError("Invalid typed assertion list")
    for row in assertions:
        if (not isinstance(row, dict)
                or set(row) != {"subject", "predicate", "value", "source_ref", "truth_mode"}
                or row["predicate"] not in PREDICATES[:3] or row["value"] not in VALUES[:-1]
                or row["truth_mode"] not in MODES
                or row["source_ref"] is not None and not isinstance(row["source_ref"], str)):
            raise ValueError("Unsupported typed assertion fields")
        seat(row["subject"])


def protected_record(record, owner, day, pending=False):
    common = {"owner", "delivery_id", "intent", "target", "recipients", "assertions", "assertion_realizations"}
    fields = common | ({"decision_id", "candidate_id", "canonical_text", "packet_hash", "send_floor_seq", "attempted_at", "status"}
                       if pending else {"accepted_seq", "original_packet_hash"})
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError("Unsupported protected record schema")
    allowed_intents = set(INTENTS[:-1]) | ({"safe_fallback"} if pending else set())
    if (record["owner"].get("session_hash") != owner["session_hash"]
            or record["owner"].get("seat") != owner["seat"]
            or type(record["owner"].get("day")) is not int or not 0 <= record["owner"]["day"] <= day
            or record["owner"].get("window_kind") != "day_vote_discussion"
            or not isinstance(record["delivery_id"], str) or not record["delivery_id"]
            or record["intent"] not in allowed_intents):
        raise ValueError("Protected record ownership/type mismatch")
    if record["target"] is not None:
        seat(record["target"])
    if (not isinstance(record["recipients"], list) or len(record["recipients"]) > 1
            or owner["seat"] in record["recipients"]):
        raise ValueError("Invalid protected audience")
    for player in record["recipients"]:
        seat(player)
    assertion_types(record["assertions"])
    if (not isinstance(record["assertion_realizations"], list)
            or len(record["assertion_realizations"]) != len(record["assertions"])
            or any(value not in {*MODES, "source_message_clarification"} for value in record["assertion_realizations"])):
        raise ValueError("Invalid protected assertion realization")
    source_hash = record["packet_hash" if pending else "original_packet_hash"]
    if not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("Missing original protected request snapshot")
    if pending:
        if (record["status"] not in {"prepared", "unknown", "accepted", "rejected"}
                or type(record["send_floor_seq"]) is not int or record["send_floor_seq"] < -1
                or isinstance(record["attempted_at"], bool)
                or not isinstance(record["attempted_at"], (float, int))
                or not math.isfinite(record["attempted_at"])):
            raise ValueError("Invalid protected pending status/clock")
        normalized_text(record["canonical_text"])
    elif record["accepted_seq"] is not None and (type(record["accepted_seq"]) is not int or record["accepted_seq"] < 0):
        raise ValueError("Invalid ACK source sequence")


def serialize(packet, candidates):
    """Return JSON-replayable slots and typed graph links, never outcome data."""
    check_packet(packet)
    scan(packet)
    scan(candidates)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 8:
        raise ValueError("Unsupported complete candidate count")
    owner, identity, window = packet["owner"], packet["identity"], packet["window"]
    actor, day = seat(identity["seat"]), identity["day"]
    if owner["seat"] != actor or window["owner"]["seat"] != actor or any(
            window["owner"].get(key) != owner[key] for key in ("session_hash", "seat")):
        raise ValueError("Slot/window ownership mismatch")
    if window["owner"].get("day") != day or window["owner"].get("window_kind") != "day_vote_discussion":
        raise ValueError("Slot window/day mismatch")
    alive = {seat(item) for item in packet["alive"]}
    facts = packet["verified_facts"]
    if len(facts) > 32:
        raise ValueError("Protected fact overflow")
    if len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate identity")
    slots, candidate_indices = [], []

    def add(kind, numbers, text, *, subject=None, speaker=None, target=None,
            recipients=(), references=(), identifier=None):
        if len(numbers) != DIMENSIONS[kind] or any(not math.isfinite(item) for item in numbers):
            raise ValueError("Typed numeric slot dimension mismatch")
        pointers = {"subject": [] if subject is None else [seat(subject)],
                    "speaker": [] if speaker is None else [seat(speaker)],
                    "target": [] if target is None else [seat(target)],
                    "recipient": [seat(player) for player in recipients]}
        clean_text, entities = normalized_text(text)
        slots.append({"type": kind, "numeric": numbers + [0.] * (32 - len(numbers)),
                      "text": clean_text, "lexical_entities": entities, "links": pointers,
                      "refs": sorted(set(references)), "id": identifier})
        return len(slots) - 1

    status = (window.get("pending") or {}).get("status", "none")
    seconds = window.get("seconds_left")
    phase = "speech" if identity["transport_phase"] == "messaging" else identity["phase"]
    numbers = (one(identity["role"], ROLES) + one(phase, ("speech", "day_vote", "night_wolf", "night_seer", "other"))
               + [bounded(day, 16), len(alive) / 7., bounded(window["accepted"], 3),
                  bounded(window["attempted"], 4), bounded(window["private_accepted"], 1),
                  bounded(window["remaining"], 5), bounded(seconds, 120) if seconds is not None else 0.,
                  float(seconds is not None)] + one(status, ("none", "prepared", "unknown", "accepted", "rejected"))
               + one(window["stage"], ("opening", "response", "commitment", "terminated"))
               + [float(window["transcript_complete"])])
    add(0, numbers, "own role phase clock and request budget", subject=actor)
    records = {seat(row["seat"]): row for row in packet["player_records"]}
    if set(records) != set(range(7)):
        raise ValueError("Exactly seven typed player records required")
    knowledge = {}
    player_indices = {}
    for player in range(7):
        local = [row for row in facts if row["subject"] == player]
        aligns = {row["alignment"] for row in local}
        roles = {row["role"] for row in local if row["role"] is not None}
        if len(aligns) > 1 or len(roles) > 1:
            raise ValueError("Contradictory protected knowledge")
        alignment, exact_role = next(iter(aligns), "unknown"), next(iter(roles), "unknown")
        knowledge[player] = alignment
        votes = [target for ballot in packet["prior_ballots"][-1:] for target in ballot["votes"].values()]
        score = packet["belief_scores"].get(str(player), 0.)
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError("Invalid uncertain belief score")
        numbers = ([float(player == actor), float(player in alive)] + one(alignment, ALIGNMENTS)
                   + one(exact_role, ("unknown", "wolf", "seer", "villager"))
                   + [float(any(row["source"] == source for row in local)) for source in SOURCES]
                   + [sum(target == player for target in votes) / 7., max(-4., min(4., score)) / 4.,
                      bounded(len(records[player]["claim_refs"]), 4), bounded(len(local), 32)])
        player_indices[player] = add(1, numbers, "persistent player knowledge", subject=player,
                                     identifier=f"player:{player}")
    seen_facts = set()
    for fact in sorted(facts, key=lambda row: (SOURCES.index(row["source"]), row["subject"], row["id"])):
        key = (fact["source"], fact["subject"], fact["alignment"], fact["role"])
        if key in seen_facts:
            continue
        seen_facts.add(key)
        numbers = one(fact["alignment"], ALIGNMENTS) + one(fact["role"] or "unknown", ("unknown", "wolf", "seer", "villager"))
        numbers += one(fact["source"], SOURCES) + [0., 0.]
        # First-known day is unavailable in B's authoritative source; record
        # unknown age explicitly rather than claiming it happened today.
        add(2, numbers, f"SUBJECT {fact['alignment']} exact-role {fact['role'] or 'unknown'} source {fact['source']}",
            subject=fact["subject"], references=[fact["id"]], identifier=fact["id"])

    claims = {row["message_ref"]: copy.deepcopy(row) for row in packet["attributed_claims"]}
    for row in packet["visible_dialogue"]:
        if row["type"] == "chat" and row["id"] not in claims:
            claims[row["id"]] = {"id": f"claim:{row['seq']}", "message_ref": row["id"],
                                  "subject_speaker": row["sender"], "channel": row["channel"],
                                  "recipients": row["recipients"], "text": row["content"],
                                  "excerpt_truncated": row.get("excerpt_truncated", False)}
    if len(claims) > 34:
        raise ValueError("Bounded source slots exceed 28 persistent plus six recent")
    latest = {}
    for row in claims.values():
        speaker = seat(row["subject_speaker"])
        latest[speaker] = max(latest.get(speaker, -1), source_sequence(row))
    for row in sorted(claims.values(), key=lambda item: (item["subject_speaker"], source_sequence(item))):
        speaker, recipients = seat(row["subject_speaker"]), row["recipients"]
        if row["channel"] != ("private" if recipients else "public"):
            raise ValueError("Source channel/audience mismatch")
        if recipients and actor != speaker and actor not in recipients:
            raise ValueError("Other-seat private source in actor packet")
        mentions = {int(match) for match in PLAYER.findall(row["text"])}
        direction = "self_sent" if speaker == actor else "self_received" if recipients else "public_other"
        numbers = ([float(speaker == actor)] + one(row["channel"], ("public", "private"))
                   + [float(speaker in alive), 0., float(actor in mentions),
                      float(source_sequence(row) == latest[speaker]), 0.,
                      float(not row.get("excerpt_truncated", False))]
                   + one(direction, ("self_sent", "self_received", "public_other")))
        add(3, numbers, row["text"], speaker=speaker, recipients=recipients,
            references=[row["id"], row["message_ref"]], identifier=row["id"])

    semantic = {}
    for order, record in enumerate(packet["self_assertions"]):
        protected_record(record, owner, day)
        if any(record["owner"].get(key) != owner[key] for key in ("session_hash", "seat")):
            raise ValueError("Other-owner ACK ledger")
        entries = list(record["assertions"])
        if record["intent"] == "coordinate_vote":
            entries.append({"subject": record["target"], "predicate": "vote_commitment", "value": "none",
                            "source_ref": None, "truth_mode": "uncertain"})
        for entry in entries:
            key = fingerprint({"assertion": entry, "recipients": record["recipients"], "intent": record["intent"]})
            if key not in semantic:
                semantic[key] = {"entry": entry, "first": record, "last": record,
                                 "first_order": order, "last_order": order, "count": 1}
            else:
                semantic[key]["last"] = record
                semantic[key]["count"] += 1
                semantic[key]["last_order"] = order
    if len(semantic) > 64:
        raise ValueError("Protected semantic commitment/bluff overflow")
    families = {}
    for key, item in semantic.items():
        entry, record = item["entry"], item["last"]
        family = (entry["predicate"], None if entry["predicate"] == "vote_commitment" else entry["subject"],
                  tuple(record["recipients"]))
        families.setdefault(family, []).append((key, item))
        item["family"] = family
    for key, item in sorted(semantic.items(), key=lambda pair: (pair[1]["first_order"], pair[0])):
        entry, record = item["entry"], item["last"]
        intent = record["intent"] if record["intent"] in INTENTS else "none"
        family = families[item["family"]]
        active = item["last_order"] == max(member["last_order"] for _, member in family)
        conflicting = len({(member["entry"]["value"], member["entry"]["subject"])
                           for _, member in family}) > 1
        numbers = (one(entry["predicate"], PREDICATES) + one(entry["value"], VALUES)
                   + one(entry["truth_mode"], MODES) + one(intent, INTENTS)
                   + one("private" if record["recipients"] else "public", ("none", "public", "private"))
                   + [float(active), float(conflicting or not active),
                      bounded(day - record["owner"]["day"], 16), bounded(record["owner"]["day"], 16)])
        add(4, numbers, f"ACKed own {intent} {entry['predicate']} {entry['value']} {entry['truth_mode']}",
            subject=entry["subject"], target=record["target"], recipients=record["recipients"],
            references=[entry["source_ref"]] if entry["source_ref"] else [], identifier=f"ack:{key}")

    def candidate_slot(candidate, pending=False):
        delivery = candidate.get("delivery", "action")
        target = candidate.get("target")
        recipients = candidate.get("recipients", [])
        assertions = candidate.get("assertions", [])
        if delivery == "action":
            kind = {"vote": "day_vote", "attack": "night_wolf", "investigate": "night_seer"}.get(
                candidate["kind"], candidate["kind"])
            source, channel = "engine", "none"
        else:
            kind = "pass" if delivery == "terminate" else "private_chat" if recipients else "public_chat"
            source = "canonical" if candidate.get("intent") in {"pass", "safe_fallback"} else "model_plan"
            channel = "none" if delivery == "terminate" else "private" if recipients else "public"
        text = candidate.get("text", candidate.get("canonical_text", ""))
        intent = candidate.get("intent") if candidate.get("intent") in INTENTS else "none"
        numbers = (one(kind, KINDS) + one(intent, INTENTS) + one(source, ("engine", "model_plan", "canonical"))
                   + [float(candidate.get("engine_legal", False))] + one(channel, ("none", "public", "private"))
                   + [float(any(row["truth_mode"] == mode for row in assertions)) for mode in MODES]
                   + one(knowledge.get(target, "unknown"), ALIGNMENTS)
                   + [bounded(len(text.split()), 50), bounded(len(assertions), 4)])
        references = [entry["source_ref"] for entry in assertions if entry["source_ref"]]
        index = add(6 if pending else 5, numbers, text or f"engine {kind}", target=target,
                    recipients=recipients, references=references, identifier=candidate.get("id", candidate.get("candidate_id")))
        # Every typed assertion subject remains a graph link, including cases
        # with two different subjects; no lexical entailment is inferred.
        slots[index]["links"]["subject"] = sorted({seat(entry["subject"]) for entry in assertions})
        return index

    legal_by_id = {row["id"]: row for row in packet["legal_actions"]}
    if any(row.get("delivery", "action") == "action" for row in candidates):
        if ({row["id"] for row in candidates} != set(legal_by_id)
                or any(row.get("delivery", "action") != "action" for row in candidates)):
            raise ValueError("Action selector must receive the complete engine-legal list")
    for candidate in candidates:
        if candidate.get("packet_hash") != packet["packet_hash"]:
            raise ValueError("Candidate belongs to a different packet")
        if candidate.get("delivery", "action") == "action":
            legal = legal_by_id.get(candidate["id"])
            if legal is None or any(candidate.get(field) != legal.get(field) for field in ("target", "kind")):
                raise ValueError("Changed engine action target/kind")
        else:
            validate_delivery_candidate(candidate, packet)
        candidate_indices.append(candidate_slot(candidate))
    pending = window.get("pending")
    if pending:
        protected_record(pending, owner, day, pending=True)
        if any(pending["owner"].get(key) != owner[key] for key in ("session_hash", "seat")):
            raise ValueError("Other-owner pending request")
        candidate_slot(dict(pending, delivery="chat"), pending=True)
    edges = []
    for recency, ballot in enumerate(reversed(packet["prior_ballots"][-3:])):
        for voter, target in ballot["votes"].items():
            if target is not None:
                edges.append((player_indices[seat(voter)], player_indices[seat(target)], 13 + recency))
    for index, row in enumerate(slots):
        for ordinal, player in enumerate(row["lexical_entities"]):
            linked = player_indices[player]
            edges.extend([(index, linked, 6 + ordinal), (linked, index, 6 + ordinal)])
        for relation, name in enumerate(RELATIONS[:4]):
            for player in row["links"][name]:
                target_index = player_indices[player]
                edges.extend([(index, target_index, relation), (target_index, index, relation)])
        for other in range(index):
            if set(row["links"]["subject"]) & set(slots[other]["links"]["subject"]):
                edges.extend([(index, other, 4), (other, index, 4)])
            if set(row["refs"]) & set(slots[other]["refs"]):
                edges.extend([(index, other, 5), (other, index, 5)])
    result = {"serializer": SERIALIZER, "owner": copy.deepcopy(owner), "packet_hash": packet["packet_hash"],
              "candidates_hash": fingerprint(candidates), "candidate_ids": [row["id"] for row in candidates],
              "candidate_indices": candidate_indices,
              "candidate_mask": [row.get("available", True) is True for row in candidates],
              "slots": slots, "edges": [list(edge) for edge in sorted(set(edges))],
              "diagnostics": {"fact_age": "unknown; source lacks first-known day",
                              "claim_age": "unknown; no inferred chronology from arbitrary seq gaps",
                              "retained_sources": len(claims), "protected_semantics": len(semantic)}}
    if not any(result["candidate_mask"]):
        raise ValueError("No available candidate")
    result["slots_hash"] = fingerprint(result)
    return result
