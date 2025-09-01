"""Aggregate testimonies with role importance scaling."""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Sequence, Any
import re

from .polarity_weights import (
    POLARITY_TABLE,
    WEIGHT_TABLE,
    FAMILY_TABLE,
    KIND_TABLE,
    TestimonyKey,
)
from .polarity import Polarity
from .dsl import RoleImportance
from .dsl_to_testimony import dispatch as dsl_dispatch
from .utils import token_to_string
try:  # pragma: no cover - allow running as script
    from ..models import Planet
except ImportError:  # pragma: no cover
    from models import Planet


def _coerce(
    testimonies: Iterable[TestimonyKey | str | RoleImportance]
) -> Tuple[Sequence[TestimonyKey | str], Dict[str, float]]:
    """Split testimonies into tokens and role importance mapping."""

    tokens: List[TestimonyKey | str] = []
    role_weights: Dict[str, float] = {}
    for raw in testimonies:
        if isinstance(raw, RoleImportance):
            name = raw.role.name.lower()
            role_weights[name] = raw.importance
            if name == "lq":
                role_weights["l7"] = raw.importance
            continue
        if isinstance(raw, TestimonyKey):
            tokens.append(raw)
            continue
        if isinstance(raw, str):
            try:
                tokens.append(TestimonyKey(raw))
            except ValueError:
                tokens.append(raw)
            continue
        try:
            tokens.append(TestimonyKey(raw))
        except ValueError:
            continue
    return tokens, role_weights


def aggregate(
    testimonies: Iterable[TestimonyKey | str | RoleImportance | Any],
    contract: Dict[str, Planet] | None = None,
) -> Tuple[float, List[Dict[str, float | TestimonyKey | Polarity | str | bool | Any]]]:
    """Aggregate testimony tokens into a score with role importance weighting."""

    raw_items: List[TestimonyKey | str | RoleImportance] = []
    extra_info: Dict[TestimonyKey | str, Dict[str, Any]] = {}
    unscored: List[Any] = []
    for raw in testimonies:
        dispatched = dsl_dispatch(raw, contract)
        if dispatched:
            for entry in dispatched:
                token = entry.get("key")
                raw_items.append(token)
                extra_info[token] = {k: v for k, v in entry.items() if k != "key"}
        else:
            raw_items.append(raw)
            unscored.append(raw)

    tokens, role_weights = _coerce(raw_items)

    total_yes = 0.0
    total_no = 0.0
    ledger: List[Dict[str, float | TestimonyKey | Polarity | str | bool | Any]] = []
    seen: set[TestimonyKey | str] = set()
    families_seen: set[str] = set()

    for token in sorted(tokens, key=token_to_string):
        if token in seen:
            continue
        seen.add(token)

        polarity = POLARITY_TABLE.get(token, Polarity.NEUTRAL) if isinstance(token, TestimonyKey) else Polarity.NEUTRAL
        if polarity is Polarity.NEUTRAL and not isinstance(token, str):
            continue

        family = FAMILY_TABLE.get(token) if isinstance(token, TestimonyKey) else None
        kind = KIND_TABLE.get(token) if isinstance(token, TestimonyKey) else None
        context_only = family is not None and family in families_seen
        if family is not None and not context_only:
            families_seen.add(family)

        weight = WEIGHT_TABLE.get(token, 0.0) if isinstance(token, TestimonyKey) else 0.0

        role_factor = 1.0
        token_name = token_to_string(token).lower()
        for role_name, factor in role_weights.items():
            pattern = rf"(^|_){re.escape(role_name)}(_|$)"
            if re.search(pattern, token_name):
                role_factor *= factor
        weight *= role_factor

        if weight < 0:
            raise ValueError("Weights must be non-negative for monotonicity")

        delta_yes = weight if (not context_only and polarity is Polarity.POSITIVE) else 0.0
        delta_no = weight if (not context_only and polarity is Polarity.NEGATIVE) else 0.0
        total_yes += delta_yes
        total_no += delta_no
        entry = {
            "key": token,
            "polarity": polarity,
            "weight": weight,
            "delta_yes": delta_yes,
            "delta_no": delta_no,
            "family": family,
            "kind": kind,
            "context": context_only,
            "role_factor": role_factor,
        }
        if token in extra_info:
            entry.update(extra_info[token])
        ledger.append(entry)

    for obj in unscored:
        ledger.append(
            {
                "key": obj,
                "polarity": Polarity.NEUTRAL,
                "weight": 0.0,
                "delta_yes": 0.0,
                "delta_no": 0.0,
                "primitive": obj,
            }
        )

    total = total_yes - total_no
    return total, ledger
