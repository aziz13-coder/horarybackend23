"""Central repository for testimony polarity and weight tables."""

from __future__ import annotations

from enum import Enum

from .polarity import Polarity
try:
    from ..rule_engine import get_rule_weight
except ImportError:  # pragma: no cover - fallback when executed as script
    from rule_engine import get_rule_weight


class TestimonyKey(Enum):
    """Canonical keys for all supported testimony tokens."""

    MOON_APPLYING_TRINE_EXAMINER_SUN = "moon_applying_trine_examiner_sun"
    MOON_APPLYING_SQUARE_EXAMINER_SUN = "moon_applying_square_examiner_sun"
    MOON_APPLYING_SEXTILE_EXAMINER_SUN = "moon_applying_sextile_examiner_sun"
    MOON_APPLYING_SEXTILE_L1 = "moon_applying_sextile_l1"
    MOON_APPLYING_SEXTILE_L7 = "moon_applying_sextile_l7"
    MOON_APPLYING_OPPOSITION_EXAMINER_SUN = "moon_applying_opposition_examiner_sun"
    MOON_APPLYING_OPPOSITION_L1 = "moon_applying_opposition_l1"
    MOON_APPLYING_OPPOSITION_L7 = "moon_applying_opposition_l7"
    L10_FORTUNATE = "l10_fortunate"
    L7_FORTUNATE = "l7_fortunate"
    L7_MALIFIC_DEBILITY = "l7_malific_debility"
    L2_FORTUNATE = "l2_fortunate"
    L2_MALIFIC_DEBILITY = "l2_malific_debility"
    L8_FORTUNATE = "l8_fortunate"
    L8_MALIFIC_DEBILITY = "l8_malific_debility"
    L5_FORTUNATE = "l5_fortunate"
    L5_MALIFIC_DEBILITY = "l5_malific_debility"
    L1_FORTUNATE = "l1_fortunate"
    L1_MALIFIC_DEBILITY = "l1_malific_debility"
    L10_MALIFIC_DEBILITY = "l10_malific_debility"
    L11_FORTUNATE = "l11_fortunate"
    L11_MALIFIC_DEBILITY = "l11_malific_debility"
    PERFECTION_DIRECT = "perfection_direct"
    PERFECTION_TRANSLATION_OF_LIGHT = "perfection_translation_of_light"
    PERFECTION_COLLECTION_OF_LIGHT = "perfection_collection_of_light"
    TRANSLATION_CONJUNCTION_WITH_RECEPTION = "translation_conjunction_with_reception"
    TRANSLATION_CONJUNCTION_WITHOUT_RECEPTION = "translation_conjunction_without_reception"
    TRANSLATION_SEXTILE_WITH_RECEPTION = "translation_sextile_with_reception"
    TRANSLATION_SEXTILE_WITHOUT_RECEPTION = "translation_sextile_without_reception"
    TRANSLATION_SQUARE_WITH_RECEPTION = "translation_square_with_reception"
    TRANSLATION_SQUARE_WITHOUT_RECEPTION = "translation_square_without_reception"
    TRANSLATION_TRINE_WITH_RECEPTION = "translation_trine_with_reception"
    TRANSLATION_TRINE_WITHOUT_RECEPTION = "translation_trine_without_reception"
    TRANSLATION_OPPOSITION_WITH_RECEPTION = "translation_opposition_with_reception"
    TRANSLATION_OPPOSITION_WITHOUT_RECEPTION = "translation_opposition_without_reception"
    COLLECTION_CONJUNCTION_WITH_RECEPTION = "collection_conjunction_with_reception"
    COLLECTION_CONJUNCTION_WITHOUT_RECEPTION = "collection_conjunction_without_reception"
    COLLECTION_SEXTILE_WITH_RECEPTION = "collection_sextile_with_reception"
    COLLECTION_SEXTILE_WITHOUT_RECEPTION = "collection_sextile_without_reception"
    COLLECTION_SQUARE_WITH_RECEPTION = "collection_square_with_reception"
    COLLECTION_SQUARE_WITHOUT_RECEPTION = "collection_square_without_reception"
    COLLECTION_TRINE_WITH_RECEPTION = "collection_trine_with_reception"
    COLLECTION_TRINE_WITHOUT_RECEPTION = "collection_trine_without_reception"
    COLLECTION_OPPOSITION_WITH_RECEPTION = "collection_opposition_with_reception"
    COLLECTION_OPPOSITION_WITHOUT_RECEPTION = "collection_opposition_without_reception"
    ESSENTIAL_DETRIMENT = "essential_detriment"
    ACCIDENTAL_RETROGRADE = "accidental_retrograde"
    SIGN_CHANGE_SUN = "sign_change_sun"
    SIGN_CHANGE_MOON = "sign_change_moon"
    SIGN_CHANGE_MERCURY = "sign_change_mercury"
    SIGN_CHANGE_VENUS = "sign_change_venus"
    SIGN_CHANGE_MARS = "sign_change_mars"
    SIGN_CHANGE_JUPITER = "sign_change_jupiter"
    SIGN_CHANGE_SATURN = "sign_change_saturn"


# Prevent pytest from collecting the enum as a test class
TestimonyKey.__test__ = False


POLARITY_TABLE: dict[TestimonyKey, Polarity] = {
    # Favorable Moon applying trine to the examiner (Sun in education questions)
    TestimonyKey.MOON_APPLYING_TRINE_EXAMINER_SUN: Polarity.POSITIVE,
    # Example negative testimony
    TestimonyKey.MOON_APPLYING_SQUARE_EXAMINER_SUN: Polarity.NEGATIVE,
    # Moon applying sextile aspects (positive)
    TestimonyKey.MOON_APPLYING_SEXTILE_EXAMINER_SUN: Polarity.POSITIVE,
    TestimonyKey.MOON_APPLYING_SEXTILE_L1: Polarity.POSITIVE,
    TestimonyKey.MOON_APPLYING_SEXTILE_L7: Polarity.POSITIVE,
    # Moon applying opposition aspects (negative)
    TestimonyKey.MOON_APPLYING_OPPOSITION_EXAMINER_SUN: Polarity.NEGATIVE,
    TestimonyKey.MOON_APPLYING_OPPOSITION_L1: Polarity.NEGATIVE,
    TestimonyKey.MOON_APPLYING_OPPOSITION_L7: Polarity.NEGATIVE,
    # Fortunate outcome promised by L10
    TestimonyKey.L10_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L7_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L7_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L2_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L2_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L8_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L8_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L5_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L5_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L1_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L1_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L10_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    TestimonyKey.L11_FORTUNATE: Polarity.POSITIVE,
    TestimonyKey.L11_MALIFIC_DEBILITY: Polarity.NEGATIVE,
    # Perfection testimonies are positive by default
    TestimonyKey.PERFECTION_DIRECT: Polarity.POSITIVE,
    TestimonyKey.PERFECTION_TRANSLATION_OF_LIGHT: Polarity.POSITIVE,
    TestimonyKey.PERFECTION_COLLECTION_OF_LIGHT: Polarity.POSITIVE,
    # Debility indicators
    TestimonyKey.ESSENTIAL_DETRIMENT: Polarity.NEGATIVE,
    TestimonyKey.ACCIDENTAL_RETROGRADE: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_SUN: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_MOON: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_MERCURY: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_VENUS: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_MARS: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_JUPITER: Polarity.NEGATIVE,
    TestimonyKey.SIGN_CHANGE_SATURN: Polarity.NEGATIVE,
}

# Mapping of tokens to rule identifiers for dynamic weight resolution
TOKEN_RULE_MAP: dict[TestimonyKey, str] = {
    TestimonyKey.MOON_APPLYING_TRINE_EXAMINER_SUN: "M1",
    TestimonyKey.MOON_APPLYING_SQUARE_EXAMINER_SUN: "M3",
    TestimonyKey.MOON_APPLYING_SEXTILE_EXAMINER_SUN: "M4",
    TestimonyKey.MOON_APPLYING_SEXTILE_L1: "M5",
    TestimonyKey.MOON_APPLYING_SEXTILE_L7: "M6",
    TestimonyKey.MOON_APPLYING_OPPOSITION_EXAMINER_SUN: "M7",
    TestimonyKey.MOON_APPLYING_OPPOSITION_L1: "M8",
    TestimonyKey.MOON_APPLYING_OPPOSITION_L7: "M9",
    TestimonyKey.L10_FORTUNATE: "LC1",
    TestimonyKey.L7_FORTUNATE: "LC2",
    TestimonyKey.L7_MALIFIC_DEBILITY: "LC3",
    TestimonyKey.L2_FORTUNATE: "LC4",
    TestimonyKey.L2_MALIFIC_DEBILITY: "LC5",
    TestimonyKey.L8_FORTUNATE: "LC6",
    TestimonyKey.L8_MALIFIC_DEBILITY: "LC7",
    TestimonyKey.L5_FORTUNATE: "LC8",
    TestimonyKey.L5_MALIFIC_DEBILITY: "LC9",
    TestimonyKey.PERFECTION_DIRECT: "P1",
    TestimonyKey.PERFECTION_TRANSLATION_OF_LIGHT: "P2",
    TestimonyKey.PERFECTION_COLLECTION_OF_LIGHT: "P3",
    TestimonyKey.ESSENTIAL_DETRIMENT: "MOD2",
    TestimonyKey.ACCIDENTAL_RETROGRADE: "MOD3",
    TestimonyKey.SIGN_CHANGE_SUN: "MOD3",
    TestimonyKey.SIGN_CHANGE_MOON: "MOD3",
    TestimonyKey.SIGN_CHANGE_MERCURY: "MOD3",
    TestimonyKey.SIGN_CHANGE_VENUS: "MOD3",
    TestimonyKey.SIGN_CHANGE_MARS: "MOD3",
    TestimonyKey.SIGN_CHANGE_JUPITER: "MOD3",
    TestimonyKey.SIGN_CHANGE_SATURN: "MOD3",
}


# ``family``/``kind`` tagging for group-based contribution control
FAMILY_TABLE: dict[TestimonyKey, str] = {
    TestimonyKey.PERFECTION_DIRECT: "perfection",
    TestimonyKey.PERFECTION_TRANSLATION_OF_LIGHT: "perfection",
    TestimonyKey.PERFECTION_COLLECTION_OF_LIGHT: "perfection",
    TestimonyKey.L7_FORTUNATE: "l7_condition",
    TestimonyKey.L7_MALIFIC_DEBILITY: "l7_condition",
    TestimonyKey.L2_FORTUNATE: "l2_condition",
    TestimonyKey.L2_MALIFIC_DEBILITY: "l2_condition",
    TestimonyKey.L8_FORTUNATE: "l8_condition",
    TestimonyKey.L8_MALIFIC_DEBILITY: "l8_condition",
    TestimonyKey.L5_FORTUNATE: "l5_condition",
    TestimonyKey.L5_MALIFIC_DEBILITY: "l5_condition",
}

KIND_TABLE: dict[TestimonyKey, str] = {
    TestimonyKey.PERFECTION_DIRECT: "direct",
    TestimonyKey.PERFECTION_TRANSLATION_OF_LIGHT: "tol",
    TestimonyKey.PERFECTION_COLLECTION_OF_LIGHT: "col",
    TestimonyKey.L7_FORTUNATE: "l7",
    TestimonyKey.L7_MALIFIC_DEBILITY: "l7",
    TestimonyKey.L2_FORTUNATE: "l2",
    TestimonyKey.L2_MALIFIC_DEBILITY: "l2",
    TestimonyKey.L8_FORTUNATE: "l8",
    TestimonyKey.L8_MALIFIC_DEBILITY: "l8",
    TestimonyKey.L5_FORTUNATE: "l5",
    TestimonyKey.L5_MALIFIC_DEBILITY: "l5",
}

# ---------------------------------------------------------------------------
# Translation/Collection token configuration
# ---------------------------------------------------------------------------

ASPECTS = ["CONJUNCTION", "SEXTILE", "SQUARE", "TRINE", "OPPOSITION"]

for aspect in ASPECTS:
    t_with = getattr(TestimonyKey, f"TRANSLATION_{aspect}_WITH_RECEPTION")
    t_without = getattr(TestimonyKey, f"TRANSLATION_{aspect}_WITHOUT_RECEPTION")
    c_with = getattr(TestimonyKey, f"COLLECTION_{aspect}_WITH_RECEPTION")
    c_without = getattr(TestimonyKey, f"COLLECTION_{aspect}_WITHOUT_RECEPTION")

    # Translation tokens
    POLARITY_TABLE[t_with] = Polarity.POSITIVE
    POLARITY_TABLE[t_without] = (
        Polarity.NEGATIVE if aspect in {"SQUARE", "OPPOSITION"} else Polarity.POSITIVE
    )
    TOKEN_RULE_MAP[t_with] = "P2"
    TOKEN_RULE_MAP[t_without] = (
        "P2_NEG" if POLARITY_TABLE[t_without] is Polarity.NEGATIVE else "P2"
    )
    FAMILY_TABLE[t_with] = FAMILY_TABLE[t_without] = "perfection"
    KIND_TABLE[t_with] = KIND_TABLE[t_without] = "tol"

    # Collection tokens
    POLARITY_TABLE[c_with] = Polarity.POSITIVE
    POLARITY_TABLE[c_without] = (
        Polarity.NEGATIVE if aspect in {"SQUARE", "OPPOSITION"} else Polarity.POSITIVE
    )
    TOKEN_RULE_MAP[c_with] = "P3"
    TOKEN_RULE_MAP[c_without] = (
        "P3_NEG" if POLARITY_TABLE[c_without] is Polarity.NEGATIVE else "P3"
    )
    FAMILY_TABLE[c_with] = FAMILY_TABLE[c_without] = "perfection"
    KIND_TABLE[c_with] = KIND_TABLE[c_without] = "col"


# Weight table is derived from rule mappings
WEIGHT_TABLE: dict[TestimonyKey, float] = {
    token: abs(get_rule_weight(rule_id)) for token, rule_id in TOKEN_RULE_MAP.items()
}

