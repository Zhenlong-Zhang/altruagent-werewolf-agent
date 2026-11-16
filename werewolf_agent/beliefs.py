"""Seat-level belief updates using only the acting seat's legal information."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


PLAYER_RE = re.compile(r"Player\s*(\d+)", re.IGNORECASE)


def seat_from_name(name: str, players: list[dict[str, Any]]) -> int | None:
    """Resolve a platform display name to its seat through the public seat map."""
    for row in players:
        if row.get("name") == name:
            try:
                return int(row["position"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def parse_role(observation: str) -> str | None:
    match = re.search(r"Your role:\s*(WOLF|SEER|VILLAGER)", observation, re.IGNORECASE)
    return match.group(1).lower() if match else None


def known_roles(observation: str, your_position: int) -> dict[int, str]:
    """Read only own-role knowledge and own seer results from this seat."""
    result: dict[int, str] = {}
    role = parse_role(observation)
    if role:
        result[your_position] = role
    if role == "wolf":
        section = re.search(r"Your fellow wolves:\s*(.*)", observation, re.IGNORECASE)
        if section:
            result.update({int(n): "wolf" for n in PLAYER_RE.findall(section.group(1))})
    if role == "seer":
        section = re.search(r"Your investigations:\s*(.*)", observation, re.IGNORECASE)
        if section:
            for seat, label in re.findall(r"Player\s*(\d+)\s+is\s+(wolf|villager)", section.group(1), re.IGNORECASE):
                result[int(seat)] = label.lower()
    return result


def build_suspicion(
    observation: str,
    game_state: dict[str, Any],
    players: list[dict[str, Any]],
    your_position: int,
    messages: list[dict[str, Any]],
) -> dict[int, float]:
    """Return a simple interpretable log-odds-like score per seat.

    Positive values mean more suspicious. Exact own-role/seer knowledge is kept
    separate in `known_roles`; revealed roles and votes are public evidence.
    """
    scores: dict[int, float] = defaultdict(float)
    known = known_roles(observation, your_position)
    for seat, role in known.items():
        scores[seat] = 8.0 if role == "wolf" else -8.0

    alive_names = set(game_state.get("alive") or [])
    alive_seats = {
        int(row["position"])
        for row in players
        if row.get("name") in alive_names and str(row.get("position", "")).isdigit()
    }

    # Dead roles are authoritative. Use them to weakly assess prior voting patterns.
    revealed: dict[int, str] = {}
    for death in game_state.get("dead") or []:
        seat = seat_from_name(str(death.get("player", "")), players)
        role = str(death.get("role", "")).lower()
        if seat is not None and role in {"wolf", "seer", "villager"}:
            revealed[seat] = role
            scores[seat] = 8.0 if role == "wolf" else -8.0

    for round_record in game_state.get("vote_history") or []:
        votes = round_record.get("votes") or {}
        eliminated = round_record.get("eliminated")
        eliminated_seat = seat_from_name(str(eliminated), players) if eliminated else None
        eliminated_role = revealed.get(eliminated_seat) if eliminated_seat is not None else None
        for voter_name, target_name in votes.items():
            voter = seat_from_name(str(voter_name), players)
            target = seat_from_name(str(target_name), players)
            if voter is None or target is None or voter in revealed:
                continue
            if eliminated_role == "wolf":
                # Voting for a confirmed wolf is weak town-lean evidence (could be bussing).
                scores[voter] -= 0.35 if target == eliminated_seat else 0.0
            elif eliminated_role == "villager" and target == eliminated_seat:
                # A villager mislynch is weak suspicion evidence, never proof.
                scores[voter] += 0.12

    # Claims and explicit accusations provide soft evidence; do not treat chat as truth.
    for msg in messages:
        sender = int(msg.get("sender", -1))
        if sender < 0 or sender not in alive_seats or sender == your_position:
            continue
        content = str(msg.get("content", ""))
        claim = re.search(r"\b(?:i am|i'm|im)\s+(?:the\s+)?seer\b", content, re.IGNORECASE)
        if claim:
            scores[sender] -= 0.18
        accused = re.search(r"Player\s*(\d+)[^\n]{0,28}\b(?:is|=|was)\s+(?:a\s+)?wolf\b", content, re.IGNORECASE)
        if accused:
            target = int(accused.group(1))
            if target in alive_seats:
                scores[target] += 0.35

    # Clamp so accumulated weak signals cannot overpower a direct investigation.
    return {seat: max(-4.0, min(4.0, float(score))) for seat, score in scores.items()}
