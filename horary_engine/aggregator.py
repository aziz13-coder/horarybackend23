"""Aggregate testimonies into a score with a contribution ledger."""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Sequence

from .polarity_weights import (
    POLARITY_TABLE,
    WEIGHT_TABLE,
    FAMILY_TABLE,
    KIND_TABLE,
    TestimonyKey,
)
from .polarity import Polarity


def _get_testimony_hierarchy_weight(token: TestimonyKey, contract: Dict[str, Any] = None) -> float:
    """Get hierarchical weight for testimony based on category rules and traditional importance."""
    
    if not contract or not hasattr(contract, 'get'):
        contract = contract or {}
    
    category_rules = contract.get('category_rules', {})
    hierarchy = category_rules.get('testimony_hierarchy', {})
    
    # Default base weight from WEIGHT_TABLE  
    base_weight = WEIGHT_TABLE.get(token, 1.0)
    
    token_name = token.value
    
    # MAJOR TESTIMONIES (Traditional horary primary indicators)
    if (token_name.startswith('perfection_') or 
        token_name.startswith('translation_') or
        token_name.startswith('collection_') or
        'mutual_reception' in token_name):
        return base_weight * 25.0  # Massive weight for primary testimonies
    
    # SECONDARY TESTIMONIES (Significator-related aspects and dignities)
    if (token_name.startswith('moon_applying_') or
        'significator' in token_name or
        token_name in ['l1_fortunate', 'l1_malific_debility']):
        
        # Check if it's relevant to primary significators
        primary_sigs = category_rules.get('primary_significators', [])
        if any(sig.lower() in token_name for sig in primary_sigs):
            return base_weight * 10.0  # High weight for primary significator conditions
        else:
            return base_weight * 5.0   # Medium weight for secondary significator conditions
    
    # MINOR TESTIMONIES (Relevant house conditions only)
    if token_name.startswith('l') and ('fortunate' in token_name or 'debility' in token_name):
        # Extract house number
        try:
            house_num = int(token_name[1:token_name.index('_')])
        except (ValueError, IndexError):
            return base_weight
            
        outcome_houses = category_rules.get('outcome_houses', [])
        irrelevant_houses = category_rules.get('irrelevant_houses', [])
        
        # CRITICAL: Filter out completely irrelevant house conditions
        if house_num in irrelevant_houses:
            return 0.0  # Zero weight for irrelevant houses
        elif house_num in outcome_houses:
            return base_weight * 2.0  # Low weight for relevant supporting houses
        else:
            return base_weight * 0.5  # Very low weight for neutral houses
    
    # CONTEXT TESTIMONIES (General conditions)  
    if (token_name in ['essential_detriment', 'accidental_retrograde'] or
        'sign_change' in token_name):
        return base_weight * 1.0  # Normal weight for context
    
    return base_weight  # Default weight for unknown testimonies


def _is_testimony_relevant(token: TestimonyKey, contract: Dict[str, Any] = None) -> bool:
    """Check if testimony is relevant to the question category."""
    
    if not contract or not hasattr(contract, 'get'):
        return True  # Include all if no contract
    
    category_rules = contract.get('category_rules', {})
    irrelevant_houses = category_rules.get('irrelevant_houses', [])
    
    token_name = token.value
    
    # Always include major testimonies
    if (token_name.startswith('perfection_') or 
        token_name.startswith('translation_') or
        token_name.startswith('collection_')):
        return True
    
    # Filter out irrelevant house conditions
    if token_name.startswith('l') and ('fortunate' in token_name or 'debility' in token_name):
        try:
            house_num = int(token_name[1:token_name.index('_')])
            return house_num not in irrelevant_houses
        except (ValueError, IndexError):
            return True
    
    return True  # Include everything else by default


def _coerce_tokens(testimonies: Iterable[TestimonyKey | str]) -> Sequence[TestimonyKey]:
    """Convert raw testimony inputs to ``TestimonyKey`` members."""

    result: List[TestimonyKey] = []
    for raw in testimonies:
        if isinstance(raw, TestimonyKey):
            result.append(raw)
        else:
            try:
                result.append(TestimonyKey(raw))
            except ValueError:
                continue
    return result


def aggregate(
    testimonies: Iterable[TestimonyKey | str],
    contract: Dict[str, Any] = None,
) -> Tuple[float, List[Dict[str, float | TestimonyKey | Polarity | str | bool]]]:
    """Aggregate testimony tokens into a weighted score and ledger.

    The aggregator is *symmetric* in that positive and negative testimonies are
    treated uniformly via the ``POLARITY_TABLE``. It enforces several
    invariants:

    * polarity: each token must map to ``Polarity.POSITIVE`` or
      ``Polarity.NEGATIVE``
    * monotonicity: weights are non-negative and contributions sum linearly
    * single contribution: duplicate tokens are ignored
    * deterministic order: processing occurs in sorted token order
    """

    total_yes = 0.0
    total_no = 0.0
    ledger: List[Dict[str, float | TestimonyKey | Polarity | str | bool]] = []
    seen: set[TestimonyKey] = set()
    families_seen: set[str] = set()

    tokens = _coerce_tokens(testimonies)
    
    # CRITICAL FIX: Filter tokens by category relevance first
    if contract:
        tokens = [token for token in tokens if _is_testimony_relevant(token, contract)]
    
    for token in sorted(tokens, key=lambda t: t.value):
        if token in seen:
            continue
        seen.add(token)
        polarity = POLARITY_TABLE.get(token, Polarity.NEUTRAL)
        if polarity is Polarity.NEUTRAL:
            continue  # unknown or neutral token

        family = FAMILY_TABLE.get(token)
        kind = KIND_TABLE.get(token)
        context_only = family is not None and family in families_seen
        if family is not None and not context_only:
            families_seen.add(family)

        # CRITICAL FIX: Use hierarchical weighting based on category and traditional importance
        if contract:
            weight = _get_testimony_hierarchy_weight(token, contract)
        else:
            weight = WEIGHT_TABLE.get(token, 0.0)
            
        if weight < 0:
            raise ValueError("Weights must be non-negative for monotonicity")
        delta_yes = weight if (not context_only and polarity is Polarity.POSITIVE) else 0.0
        delta_no = weight if (not context_only and polarity is Polarity.NEGATIVE) else 0.0
        total_yes += delta_yes
        total_no += delta_no
        ledger.append(
            {
                "key": token,
                "polarity": polarity,
                "weight": weight,
                "delta_yes": delta_yes,
                "delta_no": delta_no,
                "family": family,
                "kind": kind,
                "context": context_only,
            }
        )

    total = total_yes - total_no
    return total, ledger
