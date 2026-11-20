"""Phase-B candidate schema: explicit teacher preference, canonical delivery."""

import copy
import json

from .strategy_state import fingerprint, seat
from .strategy_candidates import (
    INTENTS, build_candidate_request as request_v1, parse_strategy_answer,
    render_assertion, render_rhetoric, validate_action_answer as action_v1,
    validate_speech_candidates as speech_v1,
)


SCHEMA = "strategy-candidates-v2"


def check_packet(packet):
    if packet.get("schema_version") != "strategy-observation-v2":
        raise ValueError("Expected owned Phase-B observation")
    if packet.get("packet_hash") != fingerprint({key: value for key, value in packet.items() if key != "packet_hash"}):
        raise ValueError("Phase-B observation snapshot changed")


def as_v1(packet):
    check_packet(packet)
    value = copy.deepcopy(packet)
    value["schema_version"] = "strategy-observation-v1"
    value.pop("packet_hash")
    value["packet_hash"] = fingerprint(value)
    return value


def build_candidate_request(packet, task):
    messages = request_v1(as_v1(packet), task)
    if task == "speech":
        messages[0]["content"] = messages[0]["content"].replace(
            '<answer>{"candidates":[three objects]}</answer>',
            '<answer>{"candidates":[three objects],"preferred_index":0}</answer>')
        messages[0]["content"] += (
            " preferred_index MUST be an integer 0, 1, or 2 identifying your actual preferred plan. "
            "Candidate order alone does not express preference. Choose the plan that best helps your faction.")
    messages[0]["content"] += " Protocol identity: strategy-candidates-v2 / typed-canonical-v1."
    messages[1]["content"] = json.dumps({"task": task, "observation": packet}, ensure_ascii=False)
    return messages


def teacher_candidates(answer, packet, task):
    view = as_v1(packet)
    if task == "action":
        choice = action_v1(answer, view)
        candidates = [dict(item, delivery="action", intent=item["kind"], packet_hash=packet["packet_hash"],
                           schema_version=SCHEMA) for item in packet["legal_actions"]]
        preferred = next(index for index, item in enumerate(candidates) if item["id"] == choice["id"])
    elif task == "speech":
        if set(answer) != {"candidates", "preferred_index"}:
            raise ValueError("Phase-B speech needs an actual teacher preferred_index")
        preferred = answer["preferred_index"]
        if type(preferred) is not int or preferred not in range(3):
            raise ValueError("Teacher preference must select one generated plan")
        candidates = speech_v1({"candidates": answer["candidates"]}, view)
        for candidate in candidates:
            candidate["packet_hash"] = packet["packet_hash"]
            candidate["schema_version"] = SCHEMA
            candidate.setdefault("assertion_realizations", [])
    else:
        raise ValueError("Unknown teacher task")
    return candidates, {"index": preferred, "candidate_id": candidates[preferred]["id"],
                        "source": "actual_native_completion_preference", "schema_version": SCHEMA}


def validate_delivery_candidate(candidate, packet):
    """Recheck typed content immediately before the RPC, not just its label."""
    check_packet(packet)
    if candidate.get("packet_hash") != packet["packet_hash"] or candidate.get("schema_version") != SCHEMA:
        raise ValueError("Candidate belongs to a different before-observation")
    common = {"id", "intent", "target", "recipients", "assertions", "assertion_realizations",
              "text", "delivery", "realization_mode", "packet_hash", "schema_version", "available", "unavailable_reason"}
    if set(candidate) - common:
        raise ValueError("Unsupported nested delivery metadata")
    target = candidate["target"]
    if target is not None and (type(target) is not int or target not in packet["alive"]):
        raise ValueError("Delivery target must be a current living seat")
    recipients = candidate["recipients"]
    if not isinstance(recipients, list) or len(recipients) > 1:
        raise ValueError("Invalid recipient list")
    recipient = seat(recipients[0]) if recipients else None
    buddies = {fact["subject"] for fact in packet["verified_facts"] if fact["source"] == "known_partner"}
    if recipient is not None and (recipient not in buddies or recipient not in packet["alive"]
                                   or packet["identity"]["role"] != "wolf"):
        raise ValueError("Delivery recipient is not a living confirmed buddy")
    assertions = candidate["assertions"]
    if not isinstance(assertions, list) or len(assertions) > 2:
        raise ValueError("Invalid delivery assertion list")
    expected_modes = ["source_message_clarification" if item.get("truth_mode") == "attributed"
                      else item.get("truth_mode") for item in assertions if isinstance(item, dict)]
    if candidate.get("assertion_realizations", []) != expected_modes or len(expected_modes) != len(assertions):
        raise ValueError("Malformed assertion realization metadata")
    if candidate["id"] == "speech:pass":
        expected = ""
        if candidate["delivery"] != "terminate" or candidate["intent"] != "pass" or target is not None or assertions or recipients:
            raise ValueError("Malformed Pass request")
    elif candidate["id"] == "speech:fallback":
        expected = "Please state your strongest evidence and intended vote."
        if candidate["delivery"] != "chat" or candidate["intent"] != "safe_fallback" or target is not None or assertions or recipients:
            raise ValueError("Malformed canonical fallback")
    else:
        if candidate["id"] not in {"speech:0", "speech:1", "speech:2"} or candidate["delivery"] != "chat":
            raise ValueError("Unsupported generated speech identity")
        if candidate["intent"] not in INTENTS or (candidate["intent"] == "coordinate_private") != bool(recipients):
            raise ValueError("Delivery intent/audience mismatch")
        bluff = any(item.get("truth_mode") == "deliberate_bluff" for item in assertions)
        if bluff != (candidate["intent"] == "bluff"):
            raise ValueError("Deliberate fiction must retain its explicit bluff intent")
        clauses = [render_assertion(item, packet, recipient) for item in assertions]
        expected = " ".join([*clauses, render_rhetoric(candidate["intent"], target, assertions)]).strip()
    if (candidate.get("realization_mode") != "typed-canonical-v1" or candidate["text"] != expected
            or len(expected) > 1024 or len(expected.split()) > 50):
        raise ValueError("Delivery text is not the validated canonical realization")
    return candidate
