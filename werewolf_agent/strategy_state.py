"""Round04 seat-owned observation compiler; no model or transport execution."""

import hashlib
import json
import math
import re

from .beliefs import build_suspicion, parse_role, seat_from_name


SCHEMA = "strategy-observation-v1"
ROLES = {"wolf", "seer", "villager"}
PLAYER = re.compile(r"\bPlayer\s*([0-6])\b", re.I)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def seat(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a player seat")
    if isinstance(value, int) or isinstance(value, str) and value.isdigit():
        result = int(value)
        if result in range(7):
            return result
    raise ValueError("Expected a player seat in 0..6")


def owned(wrapper, owner, name):
    if wrapper is None:
        return {}
    if not isinstance(wrapper, dict) or wrapper.get("owner") != owner:
        raise ValueError(f"{name} belongs to a different session/seat")
    return wrapper


def envelope(message, actor):
    value = message if isinstance(message, dict) else vars(message)
    index = value.get("seq", value.get("index"))
    if type(index) is not int or index < 0:
        raise ValueError("Visible messages require a nonnegative sequence")
    sender = seat(value.get("sender"))
    recipients = [seat(player) for player in value.get("recipients") or []]
    if len(recipients) > 1 or sender in recipients:
        raise ValueError("Invalid private-message envelope")
    if recipients and actor != sender and actor not in recipients:
        raise ValueError("Unauthorized other-seat private message")
    kind, content = value.get("type", "chat"), value.get("content", "")
    if kind not in {"chat", "terminate"} or not isinstance(content, str):
        raise ValueError("Malformed received-message envelope")
    return {"id": f"message:{index}", "seq": index, "sender": sender,
            "recipients": recipients, "channel": "private" if recipients else "public",
            "type": kind, "content": content, "move_index": value.get("move_index"),
            "day": value.get("day")}


def legal_record(action, phase):
    action_id = getattr(action, "action_id", None)
    label = getattr(action, "label", None) or ""
    payload = getattr(action, "input", {}) or {}
    if not isinstance(action_id, str) or not action_id or not isinstance(label, str) or not isinstance(payload, dict):
        raise ValueError("Malformed engine legal action")
    match = PLAYER.search(label)
    target = int(match[1]) if match else None
    explicit = payload.get("target")
    if explicit is not None:
        explicit = seat(explicit)
        if target is not None and target != explicit:
            raise ValueError("Conflicting action target metadata")
        target = explicit
    abstain = bool(re.search(r"\b(?:abstain|pass)\b", label, re.I))
    kind = "abstain" if abstain else {"day_vote": "vote", "night_wolf": "attack",
                                                   "night_seer": "investigate"}.get(phase, "engine_action")
    return {"id": action_id, "label": label, "target": target, "kind": kind,
            "engine_legal": True}


def compile_observation(state, context, history=None, window=None):
    actor = seat(context.seat_position)
    if not isinstance(context.session_id, str) or not context.session_id:
        raise ValueError("Session identifier is required for state ownership")
    owner = {"session_hash": hashlib.sha256(context.session_id.encode()).hexdigest()[:16], "seat": actor}
    history, window = owned(history, owner, "History"), owned(window, owner, "Window")
    raw = state.raw if isinstance(state.raw, dict) else {}
    game = raw.get("game_state") or {}
    players = raw.get("players") or []
    role = parse_role(state.observation)
    if role not in ROLES:
        raise ValueError("Acting player's own role is not available")
    day, phase = game.get("day"), game.get("phase")
    if type(day) is not int or day < 0 or phase not in {"day_vote", "night_wolf", "night_seer"}:
        raise ValueError("Unknown game day/phase")
    alive = []
    for name in game.get("alive") or []:
        player = seat_from_name(name, players)
        if player is None:
            raise ValueError("Cannot resolve a living player through the engine seat map")
        alive.append(seat(player))
    if len(alive) != len(set(alive)):
        raise ValueError("Duplicate living player")
    facts, alignments, exact_roles = [], {}, {}

    def fact(identifier, subject, alignment, source, exact_role=None):
        subject = seat(subject)
        if subject in alignments and alignments[subject] != alignment:
            raise ValueError("Conflicting authoritative alignment facts")
        if exact_role is not None and subject in exact_roles and exact_roles[subject] != exact_role:
            raise ValueError("Conflicting authoritative exact roles")
        alignments[subject] = alignment
        if exact_role is not None:
            exact_roles[subject] = exact_role
        record = {"id": identifier, "subject": subject, "alignment": alignment,
                  "role": exact_role, "source": source}
        previous = next((item for item in facts if item["id"] == identifier), None)
        if previous is not None and previous != record:
            raise ValueError("Conflicting repeated private check")
        if previous is None:
            facts.append(record)

    fact(f"own:{actor}", actor, "wolf" if role == "wolf" else "nonwolf", "own_role", role)
    if role == "seer":
        section = re.search(r"^Your investigations:[ \t]*([^\n]*)", state.observation, re.I | re.M)
        for subject, result in re.findall(r"Player\s*([0-6])\s+is\s+(wolf|villager)\b",
                                          section[1] if section else "", re.I):
            fact(f"check:{subject}", int(subject), "wolf" if result.lower() == "wolf" else "nonwolf", "own_check")
    if role == "wolf":
        section = re.search(r"^Your fellow wolves:[ \t]*([^\n]*)", state.observation, re.I | re.M)
        for subject in PLAYER.findall(section[1] if section else ""):
            if int(subject) != actor:
                fact(f"partner:{subject}", int(subject), "wolf", "known_partner", "wolf")
    for death in game.get("dead") or []:
        subject = seat_from_name(death.get("player"), players)
        revealed = str(death.get("role", "")).lower()
        if subject is not None and revealed in ROLES:
            fact(f"death:{subject}", subject, "wolf" if revealed == "wolf" else "nonwolf", "public_engine", revealed)

    merged = {}
    for message in [*(history.get("messages") or []), *state.new_messages]:
        row = envelope(message, actor)
        if row["seq"] in merged and merged[row["seq"]] != row:
            raise ValueError("Conflicting envelopes for one visible sequence")
        merged[row["seq"]] = row
    messages = [merged[index] for index in sorted(merged)]
    claims = [{"id": f"claim:{row['seq']}", "source": "unverified_player_chat",
               "message_ref": row["id"], "subject_speaker": row["sender"],
               "channel": row["channel"], "recipients": row["recipients"], "text": row["content"]}
              for row in messages if row["type"] == "chat"]
    ballots = []
    for index, record in enumerate(game.get("vote_history") or []):
        votes = {}
        for voter, target in (record.get("votes") or {}).items():
            voter_seat = seat_from_name(voter, players)
            if voter_seat is None:
                raise ValueError("Cannot resolve an engine ballot voter")
            votes[str(seat(voter_seat))] = seat_from_name(target, players)
        ballots.append({"id": f"ballot:{index}", "day": record.get("day"), "votes": votes,
                        "eliminated": seat_from_name(record.get("eliminated"), players)})
    # Legacy soft scores remain an uncertain channel; immutable facts are
    # projected AFTER all accusations, so repetition can never invert them.
    safe_observation = f"Your role: {role.upper()}"
    if role == "seer":
        checks = [f"Player{item['subject']} is {'wolf' if item['alignment'] == 'wolf' else 'villager'}"
                  for item in facts if item["source"] == "own_check"]
        safe_observation += "\nYour investigations: " + ("; ".join(checks) or "none")
    if role == "wolf":
        partners = [f"Player{item['subject']}" for item in facts if item["source"] == "known_partner"]
        safe_observation += "\nYour fellow wolves: " + ("; ".join(partners) or "none")
    scores = build_suspicion(safe_observation, game, players, actor, messages)
    for subject, alignment in alignments.items():
        scores[subject] = 4. if alignment == "wolf" else -4.
    legal = [legal_record(action, phase) for action in state.legal_actions]
    if len({record["id"] for record in legal}) != len(legal):
        raise ValueError("Duplicate engine legal action ID")
    accepted, attempted = window.get("accepted", 0), window.get("attempted", 0)
    if type(accepted) is not int or type(attempted) is not int or not 0 <= accepted <= 5 or attempted < accepted:
        raise ValueError("Invalid window delivery accounting")
    remaining = window.get("remaining", 5 - accepted)
    seconds = window.get("seconds_left", raw.get("messaging_seconds_left"))
    if type(remaining) is not int or not 0 <= remaining <= 5:
        raise ValueError("Invalid remaining chat quota")
    if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                                or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("Invalid phase time budget")
    pending = window.get("pending")
    if pending is not None:
        raise ValueError("Pending delivery is reserved for a typed Phase B controller; unsupported observation fields")
    packet = {
        "schema_version": SCHEMA, "owner": owner,
        "identity": {"seat": actor, "role": role, "day": day, "phase": phase,
                     "transport_phase": getattr(state, "phase", raw.get("phase", "unknown"))},
        "alive": sorted(alive), "verified_facts": facts,
        "player_records": [{"seat": player, "alignment": alignments.get(player, "unknown"),
                            "verified_refs": [item["id"] for item in facts if item["subject"] == player],
                            "claim_refs": [claim["id"] for claim in claims if claim["subject_speaker"] == player]}
                           for player in range(7)],
        "attributed_claims": claims,
        "self_assertions": [claim for claim in claims if claim["subject_speaker"] == actor],
        "visible_dialogue": messages[-20:],
        "prior_ballots": ballots,
        "belief_scores": {str(subject): value for subject, value in sorted(scores.items())},
        "legal_actions": legal,
        "window": {"owner": owner, "accepted": accepted, "attempted": attempted,
                   "remaining": remaining, "seconds_left": seconds,
                   "pending": pending, "highest_visible_seq": max(merged, default=-1),
                   "transcript_complete": window.get("transcript_complete") is True},
        "provenance": {"rules": "altruagent-seven-fixed-roles-v1", "fact_projection": "immutable-final-v1"},
    }
    packet["packet_hash"] = fingerprint(packet)
    return packet
