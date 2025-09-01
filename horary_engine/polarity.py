"""Shared polarity utilities."""
from __future__ import annotations

from enum import Enum, auto
from typing import Any


class Polarity(Enum):
    """Categorical polarity used throughout the engine."""

    POSITIVE = auto()
    NEGATIVE = auto()
    NEUTRAL = auto()


def normalize_polarity(value: Any) -> Polarity:
    """Normalize arbitrary inputs into a ``Polarity`` member."""

    if isinstance(value, Polarity):
        return value
    if isinstance(value, (int, float)):
        if value > 0:
            return Polarity.POSITIVE
        if value < 0:
            return Polarity.NEGATIVE
        return Polarity.NEUTRAL
    text = str(value).strip().lower()
    if text in {"+", "positive", "favorable"}:
        return Polarity.POSITIVE
    if text in {"-", "negative", "unfavorable"}:
        return Polarity.NEGATIVE
    return Polarity.NEUTRAL


def polarity_sign(polarity: Any) -> str:
    """Return a user-friendly sign for a polarity value.

    Neutral entries are labeled with ``"0"`` to make their absence of effect
    explicit to callers.
    """

    pol = normalize_polarity(polarity)
    if pol is Polarity.POSITIVE:
        return "+"
    if pol is Polarity.NEGATIVE:
        return "-"
    return "0"
