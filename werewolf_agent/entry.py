"""Loader entrypoint used by the official AltruAgent Python starter."""

from .policy import WerewolfPolicy


def create_agent() -> WerewolfPolicy:
    return WerewolfPolicy()
