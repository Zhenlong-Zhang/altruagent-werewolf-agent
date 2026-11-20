"""Opt-in SDK hooks: only the acting seat's existing session is queried."""

from dataclasses import replace
import time
import uuid

from altruagent import TERMINATE_MESSAGING
from altruagent.errors import AltruAgentError


def resolve(agent, state, context):
    if not getattr(agent, "requires_visible_transcript_flush", False):
        return None, context
    if context.seat_position is None:
        seat = state.raw.get("your_position")
        if type(seat) is not int or seat not in range(7):
            raise ValueError("Owned strategy transport requires its engine seat")
        context = replace(context, seat_position=seat)
    factory = getattr(agent, "strategy_transport_policy", None)
    policy = factory(context) if callable(factory) else agent
    # A population router may opt in globally while returning an unchanged
    # incumbent for this seat. Only the actual strategy policy gets hooks.
    if not getattr(policy, "requires_visible_transcript_flush", False):
        return None, context
    return policy, context


def observe(policy, state, context, now=time.monotonic):
    if policy is not None:
        policy.on_visible_update(state, context, [], "state", False, now())


def refresh(policy, game, state, context, now=time.monotonic):
    if policy is None:
        return
    try:
        result = game.get_messages(since=max(policy.memory.messages, default=-1))
        if not isinstance(result, dict) or not isinstance(result.get("messages"), list):
            raise ValueError("Malformed authorized transcript response")
        policy.on_visible_update(state, context, result["messages"], "authorized_refresh", True, now())
    except (ValueError, TypeError, OSError, RuntimeError, AltruAgentError) as exc:
        policy.on_visible_update(state, context, [], "refresh_error", False, now())
        policy.memory.record("transport_refresh_failure", error_type=type(exc).__name__)


def prepare(policy, decision, context, now=time.monotonic):
    if policy is None:
        return decision, None
    identifier = uuid.uuid4().hex
    try:
        policy.before_message_request(identifier, decision, context, now())
    except ValueError as exc:
        policy.memory.record("transport_request_overridden", delivery_id=identifier,
                             requested_kind="terminate" if decision is TERMINATE_MESSAGING else "chat",
                             reason=type(exc).__name__)
        decision = TERMINATE_MESSAGING
        policy.before_message_request(identifier, decision, context, now())
    return decision, identifier


def receipt(policy, identifier, result, state, context, now=time.monotonic):
    if policy is None:
        return
    # Merge the last terminate/chat response before the runner discards it on
    # phase transition. Its day is the pre-call window, not the next snapshot.
    rows = list(result.get("new_messages") or [])
    if isinstance(result.get("message"), dict):
        rows.append(result["message"])
    day = policy.memory.current.owner["day"]
    rows = [dict(row, day=day) for row in rows]
    policy.on_visible_update(state, context, rows, "send_result", False, now())
    policy.on_delivery_result(identifier, result, context, now())
