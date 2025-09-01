"""Utilities for turning a contribution ledger into human-readable text."""
from __future__ import annotations

from typing import List, Dict, Any

from .polarity import Polarity, polarity_sign
from .polarity_weights import TestimonyKey
from .utils import token_to_string
from .dsl import Reception, EssentialDignity, AccidentalDignity, Role

try:  # pragma: no cover - allow execution as script
    from ..models import Planet
except Exception:  # pragma: no cover
    from models import Planet


def _actor_name(actor: Any) -> str:
    """Return a readable name for either a role or planet."""

    if isinstance(actor, Role):
        return actor.name
    if isinstance(actor, Planet):
        return actor.value
    return str(actor)


def build_rationale(
    ledger: List[Dict[str, float | TestimonyKey | Polarity | Any]]
) -> List[str]:
    """Create a rationale list from a contribution ledger.

    The function is pure and does not mutate the input ledger.
    """
    result: List[str] = []
    for entry in ledger:
        primitive = entry.get("primitive") or entry.get("key")

        if isinstance(primitive, Reception):
            receiver = _actor_name(primitive.receiver)
            received = _actor_name(primitive.received)
            dignity = "sign" if primitive.dignity == "domicile" else primitive.dignity
            result.append(f"{receiver} receives {received} by {dignity}")
            continue

        if isinstance(primitive, EssentialDignity):
            actor = _actor_name(primitive.actor)
            score = primitive.score
            if isinstance(score, str):
                result.append(f"{actor} is in {score}")
            else:
                result.append(f"{actor} essential dignity {score}")
            continue

        if isinstance(primitive, AccidentalDignity):
            actor = _actor_name(primitive.actor)
            score = primitive.score
            if isinstance(score, str):
                if score.lower() == "retro":
                    result.append(f"{actor} is retrograde")
                else:
                    result.append(f"{actor} {score}")
            else:
                result.append(f"{actor} accidental dignity {score}")
            continue

        key = token_to_string(entry.get("key", ""))
        weight = entry.get("weight", 0.0)
        polarity = entry.get("polarity", Polarity.NEUTRAL)
        sign = polarity_sign(polarity)
        if sign == "0":
            result.append(f"{key} ({sign})")
        else:
            result.append(f"{key} ({sign}{weight})")
    return result
