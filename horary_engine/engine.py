# -*- coding: utf-8 -*-
"""
Complete Traditional Horary Astrology Engine with Enhanced Configuration
REFACTORED VERSION with YAML Configuration and Enhanced Moon Testimony

Created on Wed May 28 11:11:38 2025
Refactored with comprehensive configuration system and enhanced lunar calculations

@author: sabaa (enhanced with configuration system)
"""

import os
import datetime
import logging
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from types import SimpleNamespace

# Configuration system
from horary_config import get_config, cfg, HoraryError

# Timezone handling
import swisseph as swe

# Import our computational helpers
from .calculation.helpers import (
    calculate_next_station_time,
    calculate_future_longitude,
    calculate_sign_boundary_longitude,
    days_to_sign_exit,
    calculate_elongation,
    is_planet_oriental,
    sun_altitude_at_civil_twilight,
    calculate_moon_variable_speed,
    check_aspect_separation_order,
    normalize_longitude,
    degrees_to_dms,
)
from .services.geolocation import (
    TimezoneManager,
    LocationError,
    safe_geocode,
)
try:
    from ..models import Planet, Aspect, PlanetPosition, HoraryChart
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import Planet, Aspect, PlanetPosition, HoraryChart
from .dsl import (
    aspect as dsl_aspect,
    translation as dsl_translation,
    collection as dsl_collection,
    prohibition as dsl_prohibition,
    abscission as dsl_abscission,
    reception as dsl_reception,
    essential as dsl_essential,
    accidental as dsl_accidental,
    L1,
    LQ,
    Moon as MoonRole,
    Role as DslRole,
)
from .polarity_weights import TestimonyKey
from .polarity import Polarity
from .utils import token_to_string

USE_REASONING_V1 = os.getenv("USE_REASONING_V1", "").lower() in {"1", "true", "yes"}

# Setup module logger
logger = logging.getLogger(__name__)


def _structure_reasoning(reasoning: List[Any]) -> List[Dict[str, Any]]:
    """Normalize reasoning entries into structured objects.

    Existing parts of the engine build reasoning as free-form strings. To
    provide a machine-consumable format for downstream consumers (frontend,
    tests, exports), each entry is converted into a dictionary with
    ``stage``, ``rule`` and ``weight`` fields.

    Parameters
    ----------
    reasoning: list
        Original reasoning entries which may be strings or already structured
        dictionaries.

    Returns
    -------
    List[Dict[str, Any]]
        Structured reasoning entries.
    """

    structured: List[Dict[str, Any]] = []
    config = None  # lazily loaded configuration for weight lookups
    for entry in reasoning:
        if isinstance(entry, dict):
            structured.append(
                {
                    "stage": entry.get("stage", "General"),
                    "rule": entry.get("rule", entry.get("reason", "")),
                    "weight": entry.get("weight", entry.get("delta", 0)),
                }
            )
            continue

        text = str(entry).strip()
        stage = "General"
        rule = text
        weight = 0

        stage_match = re.match(r"\s*([^:]+):\s*(.*)", text)
        if stage_match:
            stage = stage_match.group(1).strip()
            rule = stage_match.group(2).strip()

        weight_match = re.search(r"\s*(?:\(([+-]?\d+)%?\)|([+-]?\d+)%)\s*$", rule)
        if weight_match:
            weight_str = weight_match.group(1) or weight_match.group(2)
            try:
                weight = int(weight_str)
            except (TypeError, ValueError):
                weight = 0
            rule = rule[: weight_match.start()].rstrip()
        else:
            # Derive implicit weights from known rule packs when possible
            if stage == "Radicality" and (
                "Ascendant too early" in rule or "Ascendant too late" in rule
            ):
                # CRITICAL FIX: ASC timing issues are confidence dampeners only, not verdict contributors
                # Don't assign negative weight that would bias toward NO verdict
                weight = 0  # No contribution to token-based verdict calculation
                rule = f"{rule} (confidence only)"

        structured.append({"stage": stage, "rule": rule, "weight": weight})

    return structured


def serialize_reasoning_v1(ledger: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Serialize a contribution ledger into standardized reasoning bundle."""

    entries: List[Dict[str, Any]] = []
    for idx, item in enumerate(ledger):
        text_source = item.get("key") or item.get("rule") or item.get("reason") or ""
        text = token_to_string(text_source)
        stage = item.get("stage") or item.get("kind")
        weight = item.get("weight", item.get("delta"))
        pol = item.get("polarity")
        if pol is None and weight is not None:
            try:
                w = float(weight)
            except (TypeError, ValueError):
                w = 0.0
            if w > 0:
                pol = Polarity.POSITIVE
            elif w < 0:
                pol = Polarity.NEGATIVE
            else:
                pol = Polarity.NEUTRAL
        if isinstance(pol, Polarity):
            polarity = pol.name.lower()
        else:
            polarity = str(pol).lower() if pol is not None else None
        entries.append(
            {
                "id": idx,
                "stage": stage,
                "text": text,
                "weight": weight,
                "polarity": polarity,
            }
        )
    return {"version": "reasoning.v1", "entries": entries}


def _evaluate_enhanced(
    scoring: List[Dict[str, Any]], category_rules: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute confidence from weighted rules with category awareness.

    Parameters
    ----------
    scoring:
        Sequence of dicts containing at minimum ``rule`` and ``weight`` keys.
        Optional ``house`` and ``factor`` keys allow filtering by the active
        category ruleset.
    category_rules:
        Rule mapping retrieved via ``get_category_rules`` describing which
        houses and factors are permitted to contribute to the score.

    Returns
    -------
    Dict[str, Any]
        ``score``     -- raw summed weight after filtering
        ``confidence``-- normalized percentage via sigmoid
        ``trace``     -- simplified list of ``{rule, weight}`` pairs actually
                          included in the calculation
    """

    allowed_houses = set(category_rules.get("outcome_houses", []))
    allowed_factors = set(category_rules.get("scored_factors", []))

    # CRITICAL DEBUG: Log testimony filtering
    print(f"TESTIMONY EVALUATION DEBUG:")
    print(f"  Allowed Houses: {allowed_houses}")
    print(f"  Allowed Factors: {allowed_factors}")
    print(f"  Input Testimonies: {len(scoring)}")
    
    total = 0.0
    trace: List[Dict[str, Any]] = []
    filtered_count = 0
    
    for entry in scoring:
        weight = float(entry.get("weight", 0) or 0)
        factor = entry.get("factor")
        house = entry.get("house")
        key = entry.get("key", "unknown")

        print(f"    Testimony: {key}, house={house}, factor={factor}, weight={weight}")
        
        if factor and allowed_factors and factor not in allowed_factors:
            print(f"      FILTERED OUT: factor '{factor}' not in allowed factors")
            filtered_count += 1
            continue
        if house and allowed_houses and house not in allowed_houses:
            print(f"      FILTERED OUT: house {house} not in allowed houses {allowed_houses}")
            filtered_count += 1
            continue

        print(f"      INCLUDED")
        trace.append({"rule": entry.get("rule", ""), "weight": weight})
        total += weight

    print(f"  Result: {len(scoring)-filtered_count}/{len(scoring)} testimonies included, total score: {total}")
    
    confidence = 1.0 / (1.0 + math.exp(-total)) * 100.0
    return {"score": total, "confidence": confidence, "trace": trace}


def extract_testimonies(chart: HoraryChart, contract: Dict[str, Planet]) -> List[Any]:
    """Extract DSL primitives from a chart using significator contract.

    This function converts raw chart calculations into the domain-specific
    language primitives defined in ``dsl.py``. It detects basic aspects along
    with higher order relationships such as translation or collection of light,
    prohibitions, receptions and planetary dignity states.

    Parameters
    ----------
    chart : HoraryChart
        Fully calculated chart.
    contract : Dict[str, Planet]
        Mapping of role names (``querent``, ``quesited``, etc.) to planets.

    Returns
    -------
    List[Any]
        DSL primitive objects describing chart conditions.
    """

    primitives: List[Any] = []

    def resolve_actor(planet: Planet):
        """Map planets to DSL actors based on the provided contract."""

        if contract.get("querent") == planet:
            return L1
        if contract.get("quesited") == planet:
            return LQ
        if planet == Planet.MOON:
            return MoonRole
        # Any additional role provided in contract
        for role_name, role_planet in (contract or {}).items():
            if role_planet == planet:
                return DslRole(role_name)
        return planet

    # ------------------------------------------------------------------
    # Aspects
    # ------------------------------------------------------------------
    for asp in getattr(chart, "aspects", []):
        actor1 = resolve_actor(asp.planet1)
        actor2 = resolve_actor(asp.planet2)
        primitives.append(
            dsl_aspect(
                actor1,
                actor2,
                asp.aspect,
                applying=asp.applying,
            )
        )

        if asp.applying and asp.aspect in (Aspect.SEXTILE, Aspect.OPPOSITION):
            if asp.planet1 == Planet.MOON or asp.planet2 == Planet.MOON:
                other = actor2 if asp.planet1 == Planet.MOON else actor1
                role_name = other.name.lower() if isinstance(other, DslRole) else ""
                if asp.aspect is Aspect.SEXTILE:
                    if role_name == "examiner":
                        primitives.append(TestimonyKey.MOON_APPLYING_SEXTILE_EXAMINER_SUN)
                    elif role_name == "l1":
                        primitives.append(TestimonyKey.MOON_APPLYING_SEXTILE_L1)
                    elif role_name in ("l7", "lq"):
                        primitives.append(TestimonyKey.MOON_APPLYING_SEXTILE_L7)
                else:  # Aspect.OPPOSITION
                    if role_name == "examiner":
                        primitives.append(TestimonyKey.MOON_APPLYING_OPPOSITION_EXAMINER_SUN)
                    elif role_name == "l1":
                        primitives.append(TestimonyKey.MOON_APPLYING_OPPOSITION_L1)
                    elif role_name in ("l7", "lq"):
                        primitives.append(TestimonyKey.MOON_APPLYING_OPPOSITION_L7)

    # ------------------------------------------------------------------
    # Dignity states (essential) and Solar Condition Testimonies
    # ------------------------------------------------------------------
    for planet, pos in getattr(chart, "planets", {}).items():
        actor = resolve_actor(planet)
        primitives.append(dsl_essential(actor, float(pos.dignity_score)))
        # Emit qualitative debility tokens for downstream rule weighting
        if pos.dignity_score <= -5:
            primitives.append(dsl_essential(actor, "detriment"))
        if pos.retrograde:
            primitives.append(dsl_accidental(actor, "retro"))
            
        # CRITICAL FIX: Generate combustion testimonies for house rulers
        if hasattr(pos, 'solar_condition') and pos.solar_condition:
            condition = pos.solar_condition.condition
            if condition == "Combustion":
                # Generate negative testimony for each house this planet rules
                for house_num, ruler in getattr(chart, "house_rulers", {}).items():
                    if ruler == planet:
                        # Create combustion testimony
                        primitives.append({
                            "key": f"l{house_num}_combust",
                            "weight": -2,  # Strong negative testimony for combustion
                            "house": int(house_num),
                            "family": f"l{house_num}_condition",
                            "kind": f"l{house_num}",
                            "polarity": "NEGATIVE",
                            "context": f"House {house_num} ruler {planet.value} combusted - matter destroyed/hidden"
                        })

    # ------------------------------------------------------------------
    # Translation, collection & prohibition
    # ------------------------------------------------------------------
    sig1 = (contract or {}).get("querent")
    sig2 = (contract or {}).get("quesited")
    if sig1 and sig2 and sig1 in chart.planets and sig2 in chart.planets:
        pos1 = chart.planets[sig1]
        pos2 = chart.planets[sig2]

        def _calc_aspect_time(
            p1: PlanetPosition,
            p2: PlanetPosition,
            aspect: Aspect,
            jd_start: float = 0.0,
            max_days: float = 0.0,
        ) -> float:
            """Return signed days until the aspect perfection.

            Positive values represent future contacts while negative values
            indicate the aspect perfected that many days in the past.
            """
            target_angles = {
                Aspect.CONJUNCTION: 0,
                Aspect.SEXTILE: 60,
                Aspect.SQUARE: 90,
                Aspect.TRINE: 120,
                Aspect.OPPOSITION: 180,
            }
            rel_speed = p1.speed - p2.speed
            if rel_speed == 0:
                return float("inf")
            diff = (
                p2.longitude
                + target_angles[aspect]
                - p1.longitude
                + 180
            ) % 360 - 180
            return diff / rel_speed

        aspect_types = [
            Aspect.CONJUNCTION,
            Aspect.SEXTILE,
            Aspect.SQUARE,
            Aspect.TRINE,
            Aspect.OPPOSITION,
        ]
        times: List[float] = []
        for a in aspect_types:
            t = _calc_aspect_time(pos1, pos2, a)
            if t and t > 0:
                times.append(t)
        if times:
            days_ahead = min(times)
            result = check_future_prohibitions(
                chart, sig1, sig2, days_ahead, _calc_aspect_time
            )
            if result.get("type") == "translation":
                primitives.append(
                    dsl_translation(
                        result["translator"],
                        sig1,
                        sig2,
                        result.get("aspect"),
                        result.get("reception", False),
                        True,
                    )
                )
            elif result.get("type") == "collection":
                primitives.append(
                    dsl_collection(
                        result["collector"],
                        sig1,
                        sig2,
                        result.get("aspect"),
                        result.get("reception", False),
                        True,
                    )
                )
            elif result.get("type") == "abscission":
                primitives.append(
                    dsl_abscission(
                        result.get("abscissor"),
                        result.get("significator"),
                        sig1 if result.get("significator") == sig2 else sig2,
                    )
                )
            elif result.get("prohibited"):
                primitives.append(
                    dsl_prohibition(
                        result.get("prohibitor"), result.get("significator")
                    )
                )

        # --------------------------------------------------------------
        # Receptions
        # --------------------------------------------------------------
        reception_calc = TraditionalReceptionCalculator()
        reception_result = reception_calc.calculate_comprehensive_reception(
            chart, sig1, sig2
        )
        for dignity in reception_result.get("planet1_receives_planet2", []):
            primitives.append(dsl_reception(sig1, sig2, dignity))
        for dignity in reception_result.get("planet2_receives_planet1", []):
            primitives.append(dsl_reception(sig2, sig1, dignity))

    # ------------------------------------------------------------------
    # House lord conditions for key topics - CATEGORY-AWARE EXTRACTION
    # ------------------------------------------------------------------
    # Determine which houses are relevant based on significator contract and category
    relevant_houses = set()
    
    # Always include houses for significators
    for house_num, ruler in chart.house_rulers.items():
        if ruler == contract.get("querent") or ruler == contract.get("quesited"):
            relevant_houses.add(house_num)
    
    # If no contract-based houses found, use category-specific houses
    if not relevant_houses:
        # Check if chart has category information
        chart_category = getattr(chart, 'category', None)
        if chart_category == "career":
            relevant_houses.update({1, 10})  # Career: querent and career/reputation
        elif chart_category in ["relationship", "lawsuit"]:
            relevant_houses.update({1, 7})  # Relationships: querent and partner
        elif chart_category == "health":
            relevant_houses.update({1, 6})  # Health: querent and health/service
        elif chart_category == "travel":
            relevant_houses.update({1, 3, 9})  # Travel: querent, short journeys, long journeys
        elif chart_category == "finance":
            relevant_houses.update({1, 2, 8})  # Finance: querent, resources, shared resources  
        elif chart_category == "education":
            relevant_houses.update({1, 3, 9})  # Education: querent, learning, higher education
        elif chart_category == "property":
            relevant_houses.update({1, 4})  # Property: querent, real estate
        else:
            # IMPROVED FALLBACK: Only include houses when planets aspect significators or are contacted
            # This prevents noise from irrelevant house conditions
            contacted_houses = set()
            significator_planets = {contract.get("querent"), contract.get("quesited")} - {None}
            
            # Find houses whose rulers are engaged in aspects
            for aspect in getattr(chart, "aspects", []):
                if aspect.applying and (aspect.planet1 in significator_planets or aspect.planet2 in significator_planets):
                    # Include houses of planets involved in significator aspects
                    for planet in [aspect.planet1, aspect.planet2]:
                        planet_pos = chart.planets.get(planet)
                        if planet_pos:
                            contacted_houses.add(planet_pos.house)
            
            # Add houses 8, 12 only if contacted (risk/hidden themes)
            relevant_houses.update(contacted_houses.intersection({2, 5, 7, 11}))
            if contacted_houses.intersection({8, 12}):
                relevant_houses.update(contacted_houses.intersection({8, 12}))
    
    # Map house numbers to testimony keys
    house_testimony_map = {
        1: (TestimonyKey.L1_FORTUNATE if hasattr(TestimonyKey, 'L1_FORTUNATE') else None, 
            TestimonyKey.L1_MALIFIC_DEBILITY if hasattr(TestimonyKey, 'L1_MALIFIC_DEBILITY') else None),
        2: (TestimonyKey.L2_FORTUNATE, TestimonyKey.L2_MALIFIC_DEBILITY),
        5: (TestimonyKey.L5_FORTUNATE, TestimonyKey.L5_MALIFIC_DEBILITY),
        7: (TestimonyKey.L7_FORTUNATE, TestimonyKey.L7_MALIFIC_DEBILITY),
        8: (TestimonyKey.L8_FORTUNATE, TestimonyKey.L8_MALIFIC_DEBILITY),
        10: (TestimonyKey.L10_FORTUNATE, TestimonyKey.L10_MALIFIC_DEBILITY if hasattr(TestimonyKey, 'L10_MALIFIC_DEBILITY') else None),
        11: (TestimonyKey.L11_FORTUNATE if hasattr(TestimonyKey, 'L11_FORTUNATE') else None,
             TestimonyKey.L11_MALIFIC_DEBILITY if hasattr(TestimonyKey, 'L11_MALIFIC_DEBILITY') else None),
    }
    
    # Only extract testimonies for relevant houses
    house_map = {house: keys for house, keys in house_testimony_map.items() 
                 if house in relevant_houses and keys[0] is not None}
    benefics = {Planet.JUPITER, Planet.VENUS}
    malefics = {Planet.MARS, Planet.SATURN}
    significator_planets = {contract.get("querent"), contract.get("quesited")} - {None}
    
    for house, (pos_key, neg_key) in house_map.items():
        ruler = chart.house_rulers.get(house)
        if not ruler:
            continue
        position = chart.planets.get(ruler)
        if not position:
            continue
            
        # Check for disqualifying conditions before assigning "fortunate" tags
        is_combust = (hasattr(position, 'solar_condition') and 
                     position.solar_condition and 
                     position.solar_condition.condition == "Combustion")
        
        is_severely_debilitated = position.dignity_score <= -8
        
        # Check if ruler applies malefic aspects to significators
        applies_malefic_to_sig = False
        for aspect in getattr(chart, "aspects", []):
            if (aspect.planet1 == ruler and aspect.planet2 in significator_planets and 
                aspect.applying and aspect.aspect in {Aspect.SQUARE, Aspect.OPPOSITION}):
                applies_malefic_to_sig = True
                break
            elif (aspect.planet2 == ruler and aspect.planet1 in significator_planets and 
                  aspect.applying and aspect.aspect in {Aspect.SQUARE, Aspect.OPPOSITION}):
                applies_malefic_to_sig = True
                break
        
        # Only assign "fortunate" if planet is not disqualified
        if (ruler in benefics or position.dignity_score >= 5) and not (is_combust or is_severely_debilitated or applies_malefic_to_sig):
            primitives.append(pos_key)
            
        # Assign malefic debility for traditional malefics, severe debilitation, or disqualifying conditions
        if (ruler in malefics or position.dignity_score <= -5 or 
            is_combust or applies_malefic_to_sig):
            primitives.append(neg_key)

    return primitives


try:
    from ..models import (
        Sign,
        SolarCondition,
        SolarAnalysis,
        AspectInfo,
        LunarAspect,
        Significator,
    )
    from ..question_analyzer import TraditionalHoraryQuestionAnalyzer
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import (
        Sign,
        SolarCondition,
        SolarAnalysis,
        AspectInfo,
        LunarAspect,
        Significator,
    )
    from question_analyzer import TraditionalHoraryQuestionAnalyzer
try:
    from ..taxonomy import (
        Category,
        resolve_category,
        resolve as resolve_significators,
        get_defaults,
    )
    from ..category_rules import get_category_rules
except ImportError:  # pragma: no cover - fallback when package context is missing
    from taxonomy import Category, resolve_category, resolve as resolve_significators, get_defaults
    from category_rules import get_category_rules
from .reception import TraditionalReceptionCalculator
from .aspects import (
    calculate_enhanced_aspects,
    calculate_moon_last_aspect,
    calculate_moon_next_aspect,
)
from .radicality import check_enhanced_radicality
from .serialization import (
    serialize_chart_for_frontend,
    serialize_lunar_aspect,
    serialize_planet_with_solar,
)
from .perfection import check_future_prohibitions


class EnhancedTraditionalAstrologicalCalculator:
    """Enhanced Traditional astrological calculations with configuration system"""
    
    def __init__(self, timezone_manager=None):
        # Set Swiss Ephemeris path
        swe.set_ephe_path('')
        
        # Initialize timezone manager (use provided or create new)
        self.timezone_manager = timezone_manager or TimezoneManager()
        
        # Traditional planets only
        self.planets_swe = {
            Planet.SUN: swe.SUN,
            Planet.MOON: swe.MOON,
            Planet.MERCURY: swe.MERCURY,
            Planet.VENUS: swe.VENUS,
            Planet.MARS: swe.MARS,
            Planet.JUPITER: swe.JUPITER,
            Planet.SATURN: swe.SATURN
        }
        
        # Traditional exaltations
        self.exaltations = {
            Planet.SUN: Sign.ARIES,
            Planet.MOON: Sign.TAURUS,
            Planet.MERCURY: Sign.VIRGO,
            Planet.VENUS: Sign.PISCES,
            Planet.MARS: Sign.CAPRICORN,
            Planet.JUPITER: Sign.CANCER,
            Planet.SATURN: Sign.LIBRA
        }
        
        # Traditional falls (opposite to exaltations)
        self.falls = {
            Planet.SUN: Sign.LIBRA,
            Planet.MOON: Sign.SCORPIO,
            Planet.MERCURY: Sign.PISCES,
            Planet.VENUS: Sign.VIRGO,
            Planet.MARS: Sign.CANCER,
            Planet.JUPITER: Sign.CAPRICORN,
            Planet.SATURN: Sign.ARIES
        }
        
        # Planets that have traditional exceptions to combustion
        self.combustion_resistant = {
            Planet.MERCURY: "Mercury rejoices near Sun",
            Planet.VENUS: "Venus as morning/evening star"
        }
    
    def get_real_moon_speed(self, jd_ut: float) -> float:
        """Get actual Moon speed from ephemeris in degrees per day"""
        try:
            moon_data, ret_flag = swe.calc_ut(jd_ut, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SPEED)
            return abs(moon_data[3])  # degrees per day
        except Exception as e:
            logger.warning(f"Failed to get Moon speed from ephemeris: {e}")
            # Fall back to configured default
            return cfg().timing.default_moon_speed_fallback
    
    def calculate_chart(self, dt_local: datetime.datetime, dt_utc: datetime.datetime, 
                       timezone_info: str, lat: float, lon: float, location_name: str) -> HoraryChart:
        """Enhanced Calculate horary chart with configuration system"""
        
        # Convert UTC datetime to Julian Day for Swiss Ephemeris
        jd_ut = swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, 
                          dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0)
        
        logger.info(f"Calculating chart for:")
        logger.info(f"  Local time: {dt_local} ({timezone_info})")
        logger.info(f"  UTC time: {dt_utc}")
        logger.info(f"  Julian Day (UT): {jd_ut}")
        # Safe logging with Unicode handling for location names
        try:
            logger.info(f"  Location: {location_name} ({lat:.4f}, {lon:.4f})")
        except UnicodeEncodeError:
            safe_location = location_name.encode('ascii', 'replace').decode('ascii')
            logger.info(f"  Location: {safe_location} ({lat:.4f}, {lon:.4f})")
        
        # Calculate traditional planets only
        planets = {}
        for planet_enum, planet_id in self.planets_swe.items():
            try:
                planet_data, ret_flag = swe.calc_ut(jd_ut, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
                
                longitude = planet_data[0]
                latitude = planet_data[1]
                speed = planet_data[3]  # degrees/day
                retrograde = speed < 0
                
                sign = self._get_sign(longitude)
                
                planets[planet_enum] = PlanetPosition(
                    planet=planet_enum,
                    longitude=longitude,
                    latitude=latitude,
                    house=0,  # Will be calculated after houses
                    sign=sign,
                    dignity_score=0,  # Will be calculated after solar analysis
                    retrograde=retrograde,
                    speed=speed
                )
                
            except Exception as e:
                logger.error(f"Error calculating {planet_enum.value}: {e}")
                # Create fallback
                planets[planet_enum] = PlanetPosition(
                    planet=planet_enum,
                    longitude=0.0,
                    latitude=0.0,
                    house=1,
                    sign=Sign.ARIES,
                    dignity_score=0,
                    speed=0.0
                )
        
        # Calculate houses (Regiomontanus - traditional for horary)
        try:
            houses_data, ascmc = swe.houses(jd_ut, lat, lon, b'R')  # Regiomontanus
            houses = list(houses_data)
            ascendant = ascmc[0]
            midheaven = ascmc[1]
        except Exception as e:
            logger.error(f"Error calculating houses: {e}")
            ascendant = 0.0
            midheaven = 90.0
            houses = [i * 30.0 for i in range(12)]
        
        # Calculate house positions and house rulers
        house_rulers = {}
        for i, cusp in enumerate(houses, 1):
            sign = self._get_sign(cusp)
            house_rulers[i] = sign.ruler
        
        # Update planet house positions
        for planet_pos in planets.values():
            house = self._calculate_house_position(planet_pos.longitude, houses)
            planet_pos.house = house
        
        # Enhanced solar condition analysis
        sun_pos = planets[Planet.SUN]
        solar_analyses = {}
        
        for planet_enum, planet_pos in planets.items():
            solar_analysis = self._analyze_enhanced_solar_condition(
                planet_enum, planet_pos, sun_pos, lat, lon, jd_ut)
            solar_analyses[planet_enum] = solar_analysis
            
            # Calculate comprehensive traditional dignity with all factors
            dignity_info = self._calculate_comprehensive_traditional_dignity(
                planet_pos.planet, planet_pos, houses, planets[Planet.SUN], solar_analysis)
            planet_pos.dignity_score = dignity_info["score"]
            planet_pos.essential_dignity = dignity_info["essential_score"]
            planet_pos.accidental_dignity = dignity_info["accidental_score"]
            planet_pos.dignities = dignity_info["dignities"]
        
        # Calculate enhanced traditional aspects
        aspects = calculate_enhanced_aspects(planets, jd_ut)

        # NEW: Calculate last and next lunar aspects
        moon_last_aspect = calculate_moon_last_aspect(
            planets, jd_ut
        )
        moon_next_aspect = calculate_moon_next_aspect(
            planets, jd_ut
        )
        
        chart = HoraryChart(
            date_time=dt_local,
            date_time_utc=dt_utc,
            timezone_info=timezone_info,
            location=(lat, lon),
            location_name=location_name,
            planets=planets,
            aspects=aspects,
            houses=houses,
            house_rulers=house_rulers,
            ascendant=ascendant,
            midheaven=midheaven,
            solar_analyses=solar_analyses,
            julian_day=jd_ut,
            moon_last_aspect=moon_last_aspect,
            moon_next_aspect=moon_next_aspect
        )
        
        return chart
    
    
    # [Continue with the rest of the methods...]
    # Due to space constraints, I'll continue with key methods
    
    def _analyze_enhanced_solar_condition(self, planet: Planet, planet_pos: PlanetPosition, 
                                        sun_pos: PlanetPosition, lat: float, lon: float,
                                        jd_ut: float) -> SolarAnalysis:
        """Enhanced solar condition analysis with configuration"""
        
        # Don't analyze the Sun itself
        if planet == Planet.SUN:
            return SolarAnalysis(
                planet=planet,
                distance_from_sun=0.0,
                condition=SolarCondition.FREE,
                exact_cazimi=False
            )
        
        # Calculate elongation
        elongation = calculate_elongation(planet_pos.longitude, sun_pos.longitude)
        
        # Use fixed traditional boundaries
        cazimi_orb = 17 / 60.0  # 0°17′
        combustion_orb = 8.5   # 8°30′
        under_beams_orb = 17.0 # 17°
        
        # Enhanced visibility check for Venus and Mercury
        traditional_exception = False
        if planet in self.combustion_resistant:
            traditional_exception = self._check_enhanced_combustion_exception(
                planet, planet_pos, sun_pos, lat, lon, jd_ut)
        
        # Determine condition by hierarchy
        if elongation <= cazimi_orb:
            # Cazimi - Heart of the Sun (maximum dignity)
            exact_cazimi = elongation <= (3/60)  # Within 3 arcminutes = exact cazimi
            return SolarAnalysis(
                planet=planet,
                distance_from_sun=elongation,
                condition=SolarCondition.CAZIMI,
                exact_cazimi=exact_cazimi,
                traditional_exception=False  # Cazimi overrides exceptions
            )
        
        elif elongation <= combustion_orb:
            # Combustion - still track traditional visibility exceptions
            return SolarAnalysis(
                planet=planet,
                distance_from_sun=elongation,
                condition=SolarCondition.COMBUSTION,
                traditional_exception=traditional_exception
            )

        elif elongation <= under_beams_orb:
            # Under the Beams - report regardless of configurability
            return SolarAnalysis(
                planet=planet,
                distance_from_sun=elongation,
                condition=SolarCondition.UNDER_BEAMS,
                traditional_exception=traditional_exception
            )
        
        # Free of solar interference
        return SolarAnalysis(
            planet=planet,
            distance_from_sun=elongation,
            condition=SolarCondition.FREE,
            traditional_exception=False
        )
    
    def _check_enhanced_combustion_exception(self, planet: Planet, planet_pos: PlanetPosition,
                                           sun_pos: PlanetPosition, lat: float, lon: float, 
                                           jd_ut: float) -> bool:
        """Enhanced combustion exception check with visibility calculations"""
        
        elongation = calculate_elongation(planet_pos.longitude, sun_pos.longitude)
        
        # Must have minimum 10° elongation
        if elongation < 10.0:
            return False
        
        # Check if planet is oriental (morning) or occidental (evening)
        is_oriental = is_planet_oriental(planet_pos.longitude, sun_pos.longitude)
        
        # Get Sun altitude at civil twilight
        sun_altitude = sun_altitude_at_civil_twilight(lat, lon, jd_ut)
        
        # Classical visibility conditions
        if planet == Planet.MERCURY:
            # Mercury rejoices near Sun but still requires the Sun to be below the horizon
            if sun_altitude <= -8.0:
                if elongation >= 10.0 and planet_pos.sign in [Sign.GEMINI, Sign.VIRGO]:
                    return True
                # Or if greater elongation (18° for Mercury)
                if elongation >= 18.0:
                    return True

        elif planet == Planet.VENUS:
            # Venus as morning/evening star exception
            if elongation >= 10.0:  # Minimum visibility
                # Check if conditions support visibility
                if sun_altitude <= -8.0:  # Civil twilight or darker
                    return True
                # Or if Venus is at maximum elongation (classical ~47°)
                if elongation >= 40.0:
                    return True
        
        return False
    
    def _calculate_enhanced_dignity(self, planet: Planet, sign: Sign, house: int, 
                                  solar_analysis: Optional[SolarAnalysis] = None) -> int:
        """Enhanced dignity calculation with configuration"""
        score = 0
        config = cfg()
        
        # Rulership
        if sign.ruler == planet:
            score += config.dignity.rulership
        
        # Exaltation
        if planet in self.exaltations and self.exaltations[planet] == sign:
            score += config.dignity.exaltation
        
        # Detriment - opposite to rulership
        detriment_signs = {
            Planet.SUN: [Sign.AQUARIUS],
            Planet.MOON: [Sign.CAPRICORN],
            Planet.MERCURY: [Sign.PISCES, Sign.SAGITTARIUS],
            Planet.VENUS: [Sign.ARIES, Sign.SCORPIO],
            Planet.MARS: [Sign.LIBRA, Sign.TAURUS],
            Planet.JUPITER: [Sign.GEMINI, Sign.VIRGO],
            Planet.SATURN: [Sign.CANCER, Sign.LEO]
        }
        
        if planet in detriment_signs and sign in detriment_signs[planet]:
            score += config.dignity.detriment
        
        # Fall
        if planet in self.falls and self.falls[planet] == sign:
            score += config.dignity.fall
        
        # House considerations - traditional joys
        house_joys = {
            Planet.MERCURY: 1,  # 1st house
            Planet.MOON: 3,     # 3rd house
            Planet.VENUS: 5,    # 5th house
            Planet.MARS: 6,     # 6th house
            Planet.SUN: 9,      # 9th house
            Planet.JUPITER: 11, # 11th house
            Planet.SATURN: 12   # 12th house
        }
        
        if planet in house_joys and house_joys[planet] == house:
            score += config.dignity.joy
        
        # ENHANCED: Use 5° rule for angularity determination
        # This requires access to houses and longitude - will be handled in calling function
        # For now, use traditional classification
        if house in [1, 4, 7, 10]:
            score += config.dignity.angular
        elif house in [2, 5, 8, 11]:  # Succedent houses
            score += config.dignity.succedent
        elif house in [3, 6, 9, 12]:  # Cadent houses
            score += config.dignity.cadent
        
        # Enhanced solar conditions
        if solar_analysis:
            condition = solar_analysis.condition
            
            if condition == SolarCondition.CAZIMI:
                # Cazimi overrides ALL negative conditions
                if solar_analysis.exact_cazimi:
                    score += config.confidence.solar.exact_cazimi_bonus
                else:
                    score += config.confidence.solar.cazimi_bonus
                    
            elif condition == SolarCondition.COMBUSTION:
                if not solar_analysis.traditional_exception:
                    score -= config.confidence.solar.combustion_penalty
                
            elif condition == SolarCondition.UNDER_BEAMS:
                if not solar_analysis.traditional_exception:
                    score -= config.confidence.solar.under_beams_penalty
        
        return score
    
    def _calculate_comprehensive_traditional_dignity(self, planet: Planet, planet_pos: PlanetPosition,
                                                   houses: List[float], sun_pos: PlanetPosition,
                                                   solar_analysis: Optional[SolarAnalysis] = None) -> Dict[str, Any]:
        """Comprehensive traditional dignity scoring with all classical factors (ENHANCED)

        Returns a dict with total score, essential/accidental breakdown, and a list of dignities present 
        so messaging and confidence adjustments can reference specific dignities like triplicity,
        term and face.
        """
        essential_score = 0
        accidental_score = 0
        dignities: List[str] = []
        config = cfg()
        sign = self._get_sign(planet_pos.longitude)
        house = planet_pos.house
        sign_degree = (planet_pos.longitude - sign.start_degree) % 30
        
        # === ESSENTIAL DIGNITIES ===
        
        # Rulership (+5)
        if sign.ruler == planet:
            essential_score += config.dignity.rulership
            dignities.append("rulership")
        
        # Exaltation (+4)
        if planet in self.exaltations and self.exaltations[planet] == sign:
            essential_score += config.dignity.exaltation
            dignities.append("exaltation")
        
        # Triplicity (+3) - traditional day/night rulers
        triplicity_score = self._calculate_triplicity_dignity(planet, sign, sun_pos)
        if triplicity_score:
            essential_score += triplicity_score
            dignities.append("triplicity")

        # Terms (+2) and Faces (+1)
        try:
            terms_table = getattr(cfg().reception.terms, sign.sign_name)
            for term in terms_table:
                if term.start <= sign_degree < term.end and Planet[term.ruler.upper()] == planet:
                    essential_score += 2
                    dignities.append("term")
                    break
        except AttributeError:
            pass

        try:
            faces_table = getattr(cfg().reception.faces, sign.sign_name)
            for face in faces_table:
                if face.start <= sign_degree < face.end and Planet[face.ruler.upper()] == planet:
                    essential_score += 1
                    dignities.append("face")
                    break
        except AttributeError:
            pass
        
        # Detriment (-5)
        detriment_signs = {
            Planet.SUN: [Sign.AQUARIUS],
            Planet.MOON: [Sign.CAPRICORN], 
            Planet.MERCURY: [Sign.PISCES, Sign.SAGITTARIUS],
            Planet.VENUS: [Sign.ARIES, Sign.SCORPIO],
            Planet.MARS: [Sign.LIBRA, Sign.TAURUS],
            Planet.JUPITER: [Sign.GEMINI, Sign.VIRGO],
            Planet.SATURN: [Sign.CANCER, Sign.LEO]
        }
        
        if planet in detriment_signs and sign in detriment_signs[planet]:
            essential_score += config.dignity.detriment
        
        # Fall (-4)
        if planet in self.falls and self.falls[planet] == sign:
            essential_score += config.dignity.fall
        
        # === ACCIDENTAL DIGNITIES ===
        
        # House joys (+2)
        house_joys = {
            Planet.MERCURY: 1, Planet.MOON: 3, Planet.VENUS: 5,
            Planet.MARS: 6, Planet.SUN: 9, Planet.JUPITER: 11, Planet.SATURN: 12
        }
        
        if planet in house_joys and house_joys[planet] == house:
            accidental_score += config.dignity.joy
        
        # Angularity with 5° rule
        angularity = self._get_traditional_angularity(planet_pos.longitude, houses, house)
        
        if angularity == "angular":
            accidental_score += config.dignity.angular
        elif angularity == "succedent":
            accidental_score += config.dignity.succedent
        else:  # cadent
            accidental_score += config.dignity.cadent
        
        # === ADVANCED TRADITIONAL FACTORS (Accidental) ===
        
        # Speed considerations
        speed_bonus = self._calculate_speed_dignity(planet, planet_pos.speed)
        accidental_score += speed_bonus
        
        # Retrograde penalty
        if planet_pos.retrograde:
            accidental_score += config.retrograde.dignity_penalty
        
        # Hayz (sect/time) bonus for planets in proper sect
        hayz_bonus = self._calculate_hayz_dignity(planet, sun_pos, houses)
        accidental_score += hayz_bonus
        
        # Solar conditions
        if solar_analysis:
            condition = solar_analysis.condition

            if condition == SolarCondition.CAZIMI:
                dignities.append("cazimi")
                if solar_analysis.exact_cazimi:
                    accidental_score += config.confidence.solar.exact_cazimi_bonus
                else:
                    accidental_score += config.confidence.solar.cazimi_bonus
            elif condition == SolarCondition.COMBUSTION:
                dignities.append("combust")
                if not solar_analysis.traditional_exception:
                    accidental_score -= config.confidence.solar.combustion_penalty
            elif condition == SolarCondition.UNDER_BEAMS:
                dignities.append("under_beams")
                if not solar_analysis.traditional_exception:
                    accidental_score -= config.confidence.solar.under_beams_penalty
        
        # Combined total score for backwards compatibility
        total_score = essential_score + accidental_score
        
        return {
            "score": total_score,
            "essential_score": essential_score,
            "accidental_score": accidental_score,
            "dignities": dignities
        }
    
    def _calculate_triplicity_dignity(self, planet: Planet, sign: Sign, sun_pos: PlanetPosition) -> int:
        """Calculate traditional triplicity dignity (ENHANCED)"""
        # Traditional triplicity rulers by element and day/night (CORRECTED)
        triplicity_rulers = {
            # Fire signs (Aries, Leo, Sagittarius) 
            Sign.ARIES: {"day": Planet.SUN, "night": Planet.JUPITER},
            Sign.LEO: {"day": Planet.SUN, "night": Planet.JUPITER},
            Sign.SAGITTARIUS: {"day": Planet.SUN, "night": Planet.JUPITER},
            
            # Earth signs (Taurus, Virgo, Capricorn)
            Sign.TAURUS: {"day": Planet.VENUS, "night": Planet.MOON},
            Sign.VIRGO: {"day": Planet.VENUS, "night": Planet.MOON},
            Sign.CAPRICORN: {"day": Planet.VENUS, "night": Planet.MOON},
            
            # Air signs (Gemini, Libra, Aquarius)
            Sign.GEMINI: {"day": Planet.SATURN, "night": Planet.MERCURY},
            Sign.LIBRA: {"day": Planet.SATURN, "night": Planet.MERCURY},
            Sign.AQUARIUS: {"day": Planet.SATURN, "night": Planet.MERCURY},
            
            # Water signs (Cancer, Scorpio, Pisces)
            Sign.CANCER: {"day": Planet.VENUS, "night": Planet.MARS},
            Sign.SCORPIO: {"day": Planet.VENUS, "night": Planet.MARS},
            Sign.PISCES: {"day": Planet.VENUS, "night": Planet.MARS}
        }
        
        if sign not in triplicity_rulers:
            return 0
            
        # Determine if it's day or night (Sun above or below horizon)
        # Day = Sun in houses 7-12 (below horizon), Night = Sun in houses 1-6 (above horizon)
        sun_house = sun_pos.house
        is_day = sun_house in [7, 8, 9, 10, 11, 12]  # Houses below horizon = day
        sect = "day" if is_day else "night"
        
        if triplicity_rulers[sign][sect] == planet:
            return cfg().dignity.triplicity  # Configurable triplicity score
            
        return 0
    
    def _calculate_speed_dignity(self, planet: Planet, speed: float) -> int:
        """Calculate dignity bonus/penalty based on planetary speed (ENHANCED)"""
        config = cfg()
        
        # Traditional fast/slow considerations
        if planet == Planet.MOON:
            if speed > 13.0:  # Fast Moon
                return config.dignity.speed_bonus
            elif speed < 11.0:  # Slow Moon  
                return config.dignity.speed_penalty
        elif planet in [Planet.MERCURY, Planet.VENUS]:
            if speed > 1.0:  # Fast inferior planets
                return config.dignity.speed_bonus
        elif planet in [Planet.MARS, Planet.JUPITER, Planet.SATURN]:
            if speed > 0.3:  # Fast superior planets
                return config.dignity.speed_bonus
            elif speed < 0.1:  # Very slow (near station)
                return config.dignity.speed_penalty
                
        return 0
    
    def _calculate_hayz_dignity(self, planet: Planet, sun_pos: PlanetPosition, houses: List[float]) -> int:
        """Calculate hayz (sect) dignity bonus (ENHANCED)"""
        config = cfg()
        
        # Determine if Sun is above horizon (day) or below (night)
        sun_house = self._calculate_house_position(sun_pos.longitude, houses)
        is_day = sun_house in [7, 8, 9, 10, 11, 12]  # Houses below horizon = day
        
        # Traditional sect assignments
        diurnal_planets = [Planet.SUN, Planet.JUPITER, Planet.SATURN]  
        nocturnal_planets = [Planet.MOON, Planet.VENUS, Planet.MARS]
        
        if planet in diurnal_planets and is_day:
            return config.dignity.hayz_bonus  # Diurnal planet in day chart
        elif planet in nocturnal_planets and not is_day:
            return config.dignity.hayz_bonus  # Nocturnal planet in night chart
        elif planet in diurnal_planets and not is_day:
            return config.dignity.hayz_penalty  # Diurnal planet in night chart
        elif planet in nocturnal_planets and is_day:
            return config.dignity.hayz_penalty  # Nocturnal planet in day chart
            
        # Mercury is neutral
        return 0
    
    
    def _calculate_enhanced_dignity_with_5degree_rule(self, planet: Planet, planet_pos: PlanetPosition, 
                                                     houses: List[float], 
                                                     solar_analysis: Optional[SolarAnalysis] = None) -> int:
        """Enhanced dignity calculation with 5° rule for angularity (ENHANCED)"""
        score = 0
        config = cfg()
        sign = self._get_sign(planet_pos.longitude)
        house = planet_pos.house
        
        # Basic dignities (same as before)
        if sign.ruler == planet:
            score += config.dignity.rulership
        
        if planet in self.exaltations and self.exaltations[planet] == sign:
            score += config.dignity.exaltation
        
        # Detriment
        detriment_signs = {
            Planet.SUN: [Sign.AQUARIUS],
            Planet.MOON: [Sign.CAPRICORN],
            Planet.MERCURY: [Sign.PISCES, Sign.SAGITTARIUS],
            Planet.VENUS: [Sign.ARIES, Sign.SCORPIO],
            Planet.MARS: [Sign.LIBRA, Sign.TAURUS],
            Planet.JUPITER: [Sign.GEMINI, Sign.VIRGO],
            Planet.SATURN: [Sign.CANCER, Sign.LEO]
        }
        
        if planet in detriment_signs and sign in detriment_signs[planet]:
            score += config.dignity.detriment
        
        if planet in self.falls and self.falls[planet] == sign:
            score += config.dignity.fall
        
        # House joys
        house_joys = {
            Planet.MERCURY: 1, Planet.MOON: 3, Planet.VENUS: 5,
            Planet.MARS: 6, Planet.SUN: 9, Planet.JUPITER: 11, Planet.SATURN: 12
        }
        
        if planet in house_joys and house_joys[planet] == house:
            score += config.dignity.joy
        
        # ENHANCED: Apply 5° rule for angularity
        angularity = self._get_traditional_angularity(planet_pos.longitude, houses, house)
        
        if angularity == "angular":
            score += config.dignity.angular
        elif angularity == "succedent":
            score += config.dignity.succedent
        else:  # cadent
            score += config.dignity.cadent
        
        # Solar conditions
        if solar_analysis:
            condition = solar_analysis.condition
            
            if condition == SolarCondition.CAZIMI:
                if solar_analysis.exact_cazimi:
                    score += config.confidence.solar.exact_cazimi_bonus
                else:
                    score += config.confidence.solar.cazimi_bonus
            elif condition == SolarCondition.COMBUSTION:
                if not solar_analysis.traditional_exception:
                    score -= config.confidence.solar.combustion_penalty
            elif condition == SolarCondition.UNDER_BEAMS:
                if not solar_analysis.traditional_exception:
                    score -= config.confidence.solar.under_beams_penalty
        
        return score
    
    
    def _get_sign(self, longitude: float) -> Sign:
        """Get zodiac sign from longitude"""
        longitude = longitude % 360
        for sign in Sign:
            if sign.start_degree <= longitude < (sign.start_degree + 30):
                return sign
        return Sign.PISCES
    
    def _calculate_house_position(self, longitude: float, houses: List[float]) -> int:
        """Calculate house position"""
        longitude = longitude % 360
        
        for i in range(12):
            current_cusp = houses[i] % 360
            next_cusp = houses[(i + 1) % 12] % 360
            
            if current_cusp > next_cusp:  # Crosses 0°
                if longitude >= current_cusp or longitude < next_cusp:
                    return i + 1
            else:
                if current_cusp <= longitude < next_cusp:
                    return i + 1
        
        return 1
    
    def _get_traditional_angularity(self, longitude: float, houses: List[float], house: int) -> str:
        """Determine traditional angularity using 5° rule (ENHANCED)"""
        longitude = longitude % 360
        
        # Get angular house cusps (1st, 4th, 7th, 10th)
        angular_cusps = [houses[0], houses[3], houses[6], houses[9]]  # 1st, 4th, 7th, 10th
        
        # Check proximity to angular cusps (5° rule)
        for cusp in angular_cusps:
            cusp = cusp % 360
            
            # Calculate minimum distance to cusp
            distance = min(
                abs(longitude - cusp),
                360 - abs(longitude - cusp)
            )
            
            if distance <= 5.0:
                return "angular"
        
        # Traditional house classification
        if house in [1, 4, 7, 10]:
            return "angular"
        elif house in [2, 5, 8, 11]:
            return "succedent"
        else:  # houses 3, 6, 9, 12
            return "cadent"


class EnhancedTraditionalHoraryJudgmentEngine:
    """Enhanced Traditional horary judgment engine with configuration system"""
    
    def __init__(self):
        self.question_analyzer = TraditionalHoraryQuestionAnalyzer()
        self.timezone_manager = TimezoneManager()
        self.calculator = EnhancedTraditionalAstrologicalCalculator(timezone_manager=self.timezone_manager)
        self.reception_calculator = TraditionalReceptionCalculator()
    
    def judge_question(self, question: str, location: str, 
                      date_str: Optional[str] = None, time_str: Optional[str] = None,
                      timezone_str: Optional[str] = None, use_current_time: bool = True,
                      manual_houses: Optional[List[int]] = None,
                      # Legacy override flags (now configurable)
                      ignore_radicality: bool = False,
                      ignore_void_moon: bool = False,
                      ignore_combustion: bool = False,
                      ignore_saturn_7th: bool = False,
                      # Legacy reception weighting (now configurable)
                      exaltation_confidence_boost: float = None) -> Dict[str, Any]:
        """Enhanced Traditional horary judgment with configuration system"""
        
        logger.info("=== JUDGE_QUESTION METHOD CALLED ===")
        logger.info(f"Location parameter: {location}")
        
        try:
            # Use configured values if not overridden
            config = cfg()
            if exaltation_confidence_boost is None:
                exaltation_confidence_boost = config.confidence.reception.mutual_exaltation_bonus
            
            # Fail-fast geocoding
            try:
                lat, lon, full_location = safe_geocode(location)
            except LocationError as e:
                raise e
            
            # Handle datetime with proper timezone support
            if use_current_time:
                dt_local, dt_utc, timezone_used = self.timezone_manager.get_current_time_for_location(lat, lon)
            else:
                if not date_str or not time_str:
                    raise ValueError("Date and time must be provided when not using current time")
                dt_local, dt_utc, timezone_used = self.timezone_manager.parse_datetime_with_timezone(
                    date_str, time_str, timezone_str, lat, lon)
            
            chart = self.calculator.calculate_chart(dt_local, dt_utc, timezone_used, lat, lon, full_location)
            
            # Analyze question traditionally
            question_analysis = self.question_analyzer.analyze_question(question)
            
            # Override with manual houses if provided
            if manual_houses:
                question_analysis["relevant_houses"] = manual_houses
                question_analysis["significators"]["quesited_house"] = manual_houses[1] if len(manual_houses) > 1 else 7
            
            # Extract window_days from timeframe analysis
            timeframe_analysis = question_analysis.get("timeframe_analysis", {})
            window_days = timeframe_analysis.get("window_days")
            
            # Use default window if no timeframe specified
            if window_days is None:
                config = cfg()
                window_days = getattr(config.timing, "default_window_days", 90)
            
            # Apply enhanced judgment with configuration
            judgment = self._apply_enhanced_judgment(
                chart, question_analysis,
                ignore_radicality, ignore_void_moon, ignore_combustion, ignore_saturn_7th,
                exaltation_confidence_boost, window_days, question_text=question)

            structured_reasoning = _structure_reasoning(judgment.get("reasoning", []))
            question_type = resolve_category(question_analysis.get("question_type"))
            category_rules = get_category_rules(question_type)
            evaluation = _evaluate_enhanced(structured_reasoning, category_rules)
            
            # Apply hybrid confidence calculation if blockers were found
            if judgment.get("hybrid_confidence_needed"):
                # For "no perfection" cases, create a synthetic blocker
                blockers = judgment["traditional_factors"].get("blockers", [])
                soften_no_perfection = False
                if not blockers and judgment["traditional_factors"].get("perfection_type") == "none":
                    # CRITICAL FIX: Calibrate severity by intent and imminent Moon→quesited contact
                    question_intent = question_analysis.get("question_intent", "REUNION")

                    # Detect imminent Moon support to the quesited (favorable, in-sign, soon)
                    try:
                        sigs = self._identify_significators(chart, question_analysis)
                        quesited_planet = sigs.get("quesited")
                    except Exception:
                        quesited_planet = None

                    moon_softener = False
                    if hasattr(chart, "moon_next_aspect") and chart.moon_next_aspect:
                        mna = chart.moon_next_aspect
                        favorable_aspects = {Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE}
                        if (
                            mna.applying and 
                            (quesited_planet is None or mna.planet == quesited_planet) and 
                            mna.aspect in favorable_aspects and 
                            getattr(mna, "perfection_eta_days", 99) <= 2.0
                        ):
                            moon_softener = True

                    # Decide severity
                    if question_intent == "SAFETY":
                        # Safety questions evaluate condition, not perfection
                        blockers = [{
                            "type": "no_perfection",
                            "severity": "warning",
                            "confidence": 65,
                            "reason": "No perfection found - evaluating condition instead"
                        }]
                        soften_no_perfection = True
                    else:
                        # Default reunion-style questions still emphasize perfection
                        severity = "fatal"
                        reason = "No viable perfection found between significators"
                        if question_intent != "REUNION" or moon_softener:
                            # Calibrate down to 'severe' for non‑reunion or when Moon applies soon to quesited
                            severity = "severe"
                            soften_no_perfection = True
                        blockers = [{
                            "type": "no_perfection",
                            "severity": severity,
                            "confidence": 85,
                            "reason": reason
                        }]
                
                # CRITICAL DEBUG: Log all inputs to hybrid confidence calculation
                print(f"HYBRID CONFIDENCE DEBUG:")
                print(f"  Question Category: {question_type}")
                print(f"  Judgment Result: {judgment['result']}")
                print(f"  Evaluation Score: {evaluation['score']}")
                print(f"  Evaluation Confidence: {evaluation['confidence']}")
                print(f"  Structured Reasoning Count: {len(structured_reasoning)}")
                print(f"  Category Rules: {category_rules}")
                print(f"  Blockers: {blockers}")
                
                # CRITICAL FIX: Use lower base confidence for SAFETY mode questions
                question_intent = question_analysis.get("question_intent", "REUNION")
                if question_intent == "SAFETY":
                    base_conf = 50  # Neutral - let evidence determine verdict
                else:
                    base_conf = 70  # Traditional bias for reunion questions
                    # Calibrate base confidence lower when softening no‑perfection
                    if soften_no_perfection:
                        # Lower base to tame extremes (target ~80–85% final on NO)
                        base_conf = 60 if moon_softener else 65
                
                hybrid_result = self._calculate_hybrid_confidence(
                    judgment["result"],
                    evaluation["score"], 
                    blockers, 
                    base_conf
                )
                
                print(f"  Hybrid Result: {hybrid_result}")
                
                judgment["confidence"] = hybrid_result["confidence"]
                judgment["confidence_breakdown"] = hybrid_result["breakdown"]
            
            # Preserve the engine's confidence when a valid perfection exists.
            # The evaluation sigmoid is useful diagnostics, but it shouldn't
            # override direct/translation/collection (or equivalent) perfection results.
            judgment["reasoning"] = structured_reasoning

            # Recommended Guardrail: If separation produced a QUALIFIED_* outcome
            # with a positive confidence, preserve that separated confidence and
            # avoid any later overwrite by evaluation fallbacks.
            preserve_separated_confidence = False
            if str(judgment.get("result")) in {"QUALIFIED_NO", "QUALIFIED YES"}:
                conf = judgment.get("confidence")
                if isinstance(conf, (int, float)) and conf > 0:
                    preserve_separated_confidence = True
                    judgment["confidence"] = int(conf)
                    judgment["confidence_source"] = "separated"

            # Determine if we have a valid perfection type
            tf = judgment.get("traditional_factors", {}) or {}
            perfection_type = tf.get("perfection_type")
            valid_perfection_types = {
                "direct",
                "direct_penalized",  # Treat penalized direct as a valid perfection subtype
                "translation",
                "collection",
                "same_ruler_unity",
                "future_house_placement",
                "moon_sun_education",
                "transaction_translation",
                # Preserve confidence for pregnancy sufficiency path
                "pregnancy_sufficiency",
            }
            
            # Valid blocking/denial factors that should also use engine/hybrid confidence
            valid_blocking_types = {
                "refranation",
                "frustration", 
                "abscission",
                "prohibition",
                "no_perfection"
            }

            if not preserve_separated_confidence:
                if judgment.get("hybrid_confidence_needed"):
                    # Already set by hybrid calc above
                    pass
                elif perfection_type in valid_perfection_types or perfection_type in valid_blocking_types:
                    # Keep engine-derived confidence for valid perfection OR blocking factors
                    engine_confidence = judgment.get("confidence")
                    if isinstance(engine_confidence, (int, float)) and engine_confidence > 0:
                        judgment["confidence"] = int(engine_confidence)
                        judgment.setdefault("confidence_source", "engine")
                    else:
                        # CRITICAL FIX: For blocking factors, still use hybrid confidence instead of evaluation
                        if perfection_type in valid_blocking_types:
                            # Use hybrid confidence for blocking factors
                            hybrid_result = self._calculate_hybrid_confidence(
                                judgment["result"],
                                evaluation["score"], 
                                [], 
                                50  # Lower base confidence for blocking factors
                            )
                            judgment["confidence"] = hybrid_result["confidence"]
                            judgment["confidence_source"] = "hybrid_blocking"
                            print(f"  BLOCKING FACTOR HYBRID: Used hybrid confidence for {perfection_type}: {hybrid_result['confidence']}%")
                        else:
                            # Use evaluation confidence for missing perfection confidence
                            fallback_confidence = int(evaluation["confidence"])
                            print(f"  CONFIDENCE OVERRIDE: Engine confidence was {engine_confidence}, using evaluation: {fallback_confidence}%")
                            judgment["confidence"] = fallback_confidence
                            judgment["confidence_source"] = "evaluation"
                else:
                    # Fall back to evaluation-derived confidence when no perfection
                    judgment["confidence"] = int(evaluation["confidence"])
                    judgment["confidence_source"] = "evaluation"

            judgment["scoring_trace"] = evaluation["trace"]
            reasoning_bundle = serialize_reasoning_v1(structured_reasoning) if USE_REASONING_V1 else None

            # Serialize chart data for frontend
            chart_data_serialized = serialize_chart_for_frontend(chart, chart.solar_analyses)

            general_info = self._calculate_general_info(chart)
            considerations = self._calculate_considerations(chart, question_analysis)

            return {
                "question": question,
                "judgment": judgment["result"],
                "confidence": judgment["confidence"],
                "reasoning": judgment["reasoning"],
                "scoring_trace": judgment.get("scoring_trace", []),
                **({"reasoning_v1": reasoning_bundle} if reasoning_bundle is not None else {}),

                "chart_data": chart_data_serialized,
                
                "question_analysis": question_analysis,
                "timing": judgment.get("timing"),
                "moon_aspects": self._build_moon_story(chart),  # Enhanced Moon story
                "traditional_factors": judgment.get("traditional_factors", {}),
                "solar_factors": judgment.get("solar_factors", {}),
                "general_info": general_info,
                "considerations": considerations,
                
                # NEW: Enhanced lunar aspects
                "moon_last_aspect": serialize_lunar_aspect(chart.moon_last_aspect),
                "moon_next_aspect": serialize_lunar_aspect(chart.moon_next_aspect),
                
                "timezone_info": {
                    "local_time": dt_local.isoformat(),
                    "utc_time": dt_utc.isoformat(),
                    "timezone": timezone_used,
                    "location_name": full_location,
                    "coordinates": {
                        "latitude": lat,
                        "longitude": lon
                    }
                }
            }
            
        except LocationError as e:
            return {
                "error": str(e),
                "judgment": "LOCATION_ERROR",
                "confidence": 0,
                "reasoning": _structure_reasoning([f"Location error: {e}"]),
                "error_type": "LocationError"
            }
        except Exception as e:
            import traceback
            logger.error(f"Error in judge_question: {e}")
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "judgment": "ERROR",
                "confidence": 0,
                "reasoning": _structure_reasoning([f"Calculation error: {e}"])
            }
    
    def _moon_aspects_significator_directly(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> bool:
        """
        HELPER: Check if Moon's next aspect is directly to a significator
        
        This helps identify when Moon aspects are being used as false perfection
        instead of proper significator-to-significator perfection.
        """
        if chart.moon_next_aspect:
            target_planet = chart.moon_next_aspect.planet
            return target_planet in [querent, quesited]
        return False
    
    
    # NEW: Enhanced Moon accidental dignity helpers
    def _moon_phase_bonus(self, chart: HoraryChart) -> int:
        """Calculate Moon phase bonus from configuration"""
        
        moon_pos = chart.planets[Planet.MOON]
        sun_pos = chart.planets[Planet.SUN]
        
        # Calculate angular distance (elongation)
        elongation = abs(moon_pos.longitude - sun_pos.longitude)
        if elongation > 180:
            elongation = 360 - elongation
        
        config = cfg()
        
        # Determine phase and return bonus
        if 0 <= elongation < 30:
            return config.moon.phase_bonus.new_moon
        elif 30 <= elongation < 60:
            return config.moon.phase_bonus.waxing_crescent
        elif 60 <= elongation < 120:
            return config.moon.phase_bonus.first_quarter
        elif 120 <= elongation < 150:
            return config.moon.phase_bonus.waxing_gibbous
        elif 150 <= elongation < 210:
            return config.moon.phase_bonus.full_moon
        elif 210 <= elongation < 240:
            return config.moon.phase_bonus.waning_gibbous
        elif 240 <= elongation < 300:
            return config.moon.phase_bonus.last_quarter
        else:  # 300 <= elongation < 360
            return config.moon.phase_bonus.waning_crescent
    
    def _moon_speed_bonus(self, chart: HoraryChart) -> int:
        """Calculate Moon speed bonus from configuration"""
        
        moon_speed = abs(chart.planets[Planet.MOON].speed)
        config = cfg()
        
        if moon_speed < 11.0:
            return config.moon.speed_bonus.very_slow
        elif moon_speed < 12.0:
            return config.moon.speed_bonus.slow
        elif moon_speed < 14.0:
            return config.moon.speed_bonus.average
        elif moon_speed < 15.0:
            return config.moon.speed_bonus.fast
        else:
            return config.moon.speed_bonus.very_fast
    
    def _moon_angularity_bonus(self, chart: HoraryChart) -> int:
        """Calculate Moon angularity bonus from configuration"""
        
        moon_house = chart.planets[Planet.MOON].house
        config = cfg()
        
        if moon_house in [1, 4, 7, 10]:
            return config.moon.angularity_bonus.angular
        elif moon_house in [2, 5, 8, 11]:
            return config.moon.angularity_bonus.succedent
        else:  # cadent houses 3, 6, 9, 12
            return config.moon.angularity_bonus.cadent

    # ---------------- General Info Helpers -----------------

    PLANET_SEQUENCE = [
        Planet.SATURN,
        Planet.JUPITER,
        Planet.MARS,
        Planet.SUN,
        Planet.VENUS,
        Planet.MERCURY,
        Planet.MOON,
    ]

    PLANETARY_DAY_RULERS = {
        0: Planet.MOON,      # Monday
        1: Planet.MARS,      # Tuesday
        2: Planet.MERCURY,   # Wednesday
        3: Planet.JUPITER,   # Thursday
        4: Planet.VENUS,     # Friday
        5: Planet.SATURN,    # Saturday
        6: Planet.SUN        # Sunday
    }

    LUNAR_MANSIONS = [
        "Al Sharatain", "Al Butain", "Al Thurayya", "Al Dabaran",
        "Al Hak'ah", "Al Han'ah", "Al Dhira", "Al Nathrah",
        "Al Tarf", "Al Jabhah", "Al Zubrah", "Al Sarfah",
        "Al Awwa", "Al Simak", "Al Ghafr", "Al Jubana",
        "Iklil", "Al Qalb", "Al Shaula", "Al Na'am",
        "Al Baldah", "Sa'd al Dhabih", "Sa'd Bula", "Sa'd al Su'ud",
        "Sa'd al Akhbiya", "Al Fargh al Mukdim", "Al Fargh al Thani",
        "Batn al Hut"
    ]

    def _get_moon_phase_name(self, chart: HoraryChart) -> str:
        """Return textual Moon phase name"""
        moon_pos = chart.planets[Planet.MOON]
        sun_pos = chart.planets[Planet.SUN]

        elongation = abs(moon_pos.longitude - sun_pos.longitude)
        if elongation > 180:
            elongation = 360 - elongation

        if 0 <= elongation < 30:
            return "New Moon"
        elif 30 <= elongation < 60:
            return "Waxing Crescent"
        elif 60 <= elongation < 120:
            return "First Quarter"
        elif 120 <= elongation < 150:
            return "Waxing Gibbous"
        elif 150 <= elongation < 210:
            return "Full Moon"
        elif 210 <= elongation < 240:
            return "Waning Gibbous"
        elif 240 <= elongation < 300:
            return "Last Quarter"
        else:
            return "Waning Crescent"

    def _moon_speed_category(self, speed: float) -> str:
        """Return a text category for Moon's speed"""
        speed = abs(speed)
        if speed < 11.0:
            return "Very Slow"
        elif speed < 12.0:
            return "Slow"
        elif speed < 14.0:
            return "Average"
        elif speed < 15.0:
            return "Fast"
        else:
            return "Very Fast"

    def _calculate_general_info(self, chart: HoraryChart) -> Dict[str, Any]:
        """Calculate general chart information for frontend display"""
        dt_local = chart.date_time
        weekday = dt_local.weekday()
        day_ruler = self.PLANETARY_DAY_RULERS.get(weekday, Planet.SUN)

        hour_index = dt_local.hour
        start_idx = self.PLANET_SEQUENCE.index(day_ruler)
        hour_ruler = self.PLANET_SEQUENCE[(start_idx + hour_index) % 7]

        moon_pos = chart.planets[Planet.MOON]

        mansion_index = int((moon_pos.longitude % 360) / (360 / 28)) + 1
        mansion_name = self.LUNAR_MANSIONS[mansion_index - 1]

        void_info = self._is_moon_void_of_course_enhanced(chart)

        return {
            "planetary_day": day_ruler.value,
            "planetary_hour": hour_ruler.value,
            "moon_phase": self._get_moon_phase_name(chart),
            "moon_mansion": {
                "number": mansion_index,
                "name": mansion_name,
            },
            "moon_condition": {
                "sign": moon_pos.sign.sign_name,
                "speed": moon_pos.speed,
                "speed_category": self._moon_speed_category(moon_pos.speed),
                "void_of_course": void_info["void"],
                "void_reason": void_info["reason"],
            }
        }

    def _calculate_considerations(self, chart: HoraryChart, question_analysis: Dict) -> Dict[str, Any]:
        """Return standard horary considerations"""
        radicality = check_enhanced_radicality(chart)
        moon_void = self._is_moon_void_of_course_enhanced(chart)

        return {
            "radical": radicality["valid"],
            "radical_reason": radicality["reason"],
            "moon_void": moon_void["void"],
            "moon_void_reason": moon_void["reason"],
        }
    
    # [Continue with rest of enhanced methods...]
    # Due to space constraints, I'll highlight the key enhanced methods
    
    def _classify_question_intent(self, question: str, category: str) -> str:
        """Classify if question asks for OCCURRENCE ('will it happen?') or QUALITY ('is it favorable?')"""
        
        if not question:
            return "OCCURRENCE"  # Default fallback
            
        question_lower = question.lower().strip()
        
        # QUALITY patterns - asking for evaluation/advisability
        quality_indicators = [
            'should i', 'should we', 'is it good', 'is it wise', 'is it favorable',
            'is it advisable', 'is it beneficial', 'is it worth', 'is it right',
            'good idea', 'wise to', 'advisable to', 'beneficial to', 'favorable',
            'worth it', 'good for me', 'good for us', 'right choice', 'best option'
        ]
        
        # Check for explicit quality patterns
        if any(indicator in question_lower for indicator in quality_indicators):
            return "QUALITY"
        
        # OCCURRENCE patterns - asking for likelihood/possibility
        occurrence_indicators = [
            'will i', 'will we', 'will it', 'will this', 'will there',
            'can i', 'can we', 'can it', 'going to', 'likely to',
            'happen', 'succeed', 'work out', 'come back', 'return',
            'get', 'find', 'recover', 'obtain', 'achieve'
        ]
        
        # Check for explicit occurrence patterns
        if any(indicator in question_lower for indicator in occurrence_indicators):
            return "OCCURRENCE"
        
        # Category-based classification for ambiguous questions
        from taxonomy import Category, resolve_category
        cat = resolve_category(category)
        
        # Categories that typically ask occurrence questions
        if cat in [Category.LOST_OBJECT, Category.PREGNANCY, Category.DEATH, Category.FRIEND_ENEMY]:
            return "OCCURRENCE"  # "Will I find it?" "Will I conceive?" etc.
            
        # Categories that typically ask quality questions  
        elif cat in [Category.MARRIAGE, Category.CAREER, Category.FUNDING, Category.PROPERTY]:
            return "QUALITY"     # "Is this marriage good?" "Should I take this job?" etc.
        
        # Travel can be either - but lean towards occurrence
        elif cat == Category.TRAVEL:
            if any(word in question_lower for word in ['should', 'good', 'wise', 'advisable']):
                return "QUALITY"
            else:
                return "OCCURRENCE"
                
        return "OCCURRENCE"  # Default fallback
    
    def _apply_enhanced_judgment(self, chart: HoraryChart, question_analysis: Dict,
                               ignore_radicality: bool = False, ignore_void_moon: bool = False,
                               ignore_combustion: bool = False, ignore_saturn_7th: bool = False,
                               exaltation_confidence_boost: float = 15.0, window_days: int = None, 
                               question_text: str = "") -> Dict[str, Any]:
        """Enhanced judgment with configuration system"""
        
        reasoning = []
        # Prevent duplicate Moon supportive note when appended from multiple branches
        added_moon_support_note = False
        config = cfg()
        question_type = resolve_category(question_analysis.get("question_type"))
        category_rules = get_category_rules(question_type)
        
        # CRITICAL FIX: Classify question intent to provide focused answers
        # Use the original question text passed as parameter, not from question_analysis
        question_intent = self._classify_question_intent(question_text, question_analysis.get("question_type", ""))
        
        reasoning.append({
            "stage": "Question Classification",
            "rule": f"Question classified as {question_intent}: {question_text[:50]}{'...' if len(question_text) > 50 else ''}",
            "weight": 0
        })
        
        # Initialize evidence ledger
        evidence_ledger = {
            "base_confidence": config.confidence.base_confidence,
            "rules_applied": [],
            "deltas": [],
            "caps": {},
            "final_confidence": 0,
            "verdict": "PENDING"
        }
        confidence = config.confidence.base_confidence
        separated_minimum = 0  # Initialize - will be set properly if separation logic is used
        
        def add_evidence(rule_name: str, delta: float, reason: str, rule_type: str = "modifier"):
            """Helper to add evidence to ledger"""
            evidence_ledger["rules_applied"].append(rule_name)
            evidence_ledger["deltas"].append({
                "rule": rule_name,
                "delta": delta,
                "reason": reason,
                "type": rule_type
            })
            return delta
        
        def apply_confidence_penalty(current_conf: float, penalty: int, reason: str, traditional_minimum: int = 10) -> float:
            """Apply confidence penalty while respecting separated confidence minimum"""
            # Use the higher of traditional minimum or separated minimum (if set)
            effective_minimum = max(traditional_minimum, separated_minimum)
            new_confidence = max(effective_minimum, current_conf - penalty)
            return new_confidence
        asc_penalty = 0
        void_penalty = 0
        
        # 1. Enhanced radicality with configuration
        if not ignore_radicality:
            radicality = check_enhanced_radicality(chart, ignore_saturn_7th)
            if not radicality["valid"]:
                reasoning.append(f"Radicality: {radicality['reason']}")
                reason = radicality["reason"]
                # CRITICAL FIX: ASC timing issues affect confidence only, not verdict
                if "Ascendant too early" in reason or "Ascendant too late" in reason:
                    # Late/early ASC should be confidence dampener only, per traditional usage
                    # Don't force verdict to NO - let testimony evaluation determine outcome
                    asc_penalty = getattr(config.radicality, "asc_warning_penalty", 15)
                    # CRITICAL FIX: Never allow ASC penalty to zero confidence completely
                    # Preserve minimum for later quality vs occurrence assessment
                    confidence = max(confidence - asc_penalty, 25)
                    # Continue to full analysis instead of returning NO verdict
                else:
                    # Other radicality issues are more serious
                    if getattr(config.radicality, "gating", False):
                        return {
                            "result": "NO", 
                            "confidence": min(confidence, config.confidence.lunar_confidence_caps.neutral),
                            "reasoning": reasoning,
                            "timing": None,
                        }
                    confidence = min(confidence, config.confidence.lunar_confidence_caps.neutral)
            else:
                reasoning.append(f"Radicality: {radicality['reason']}")
        else:
            reasoning.append("Radicality: Bypassed by override (chart validity check disabled)")

        # 1.5. Void-of-Course Moon caution (no early return)
        if not ignore_void_moon:
            void_check = self._is_moon_void_of_course_enhanced(chart)
            if void_check["void"]:
                # CRITICAL FIX: Adjust VOC penalty based on sign mitigation
                base_void_penalty = getattr(config.moon, "void_penalty", 10)
                
                if void_check.get("sign_mitigation"):
                    # Milder penalty for Lilly-favorable signs
                    void_penalty = max(base_void_penalty // 2, 3)  # Reduce by half, minimum 3
                else:
                    void_penalty = base_void_penalty
                
                if void_check.get("exception") or void_check.get("sign_mitigation"):
                    reasoning.append(
                        {
                            "stage": "Moon",
                            "rule": f"Void Moon noted but excepted: {void_check['reason']}",
                            "weight": -void_penalty,
                        }
                    )
                else:
                    override_check = TraditionalOverrides.check_void_moon_overrides(
                        chart, question_analysis, self
                    )
                    if override_check.get("can_override"):
                        reasoning.append(
                            {
                                "stage": "Moon",
                                "rule": f"Void Moon noted but overridden: {override_check['reason']}",
                                "weight": -void_penalty,
                            }
                        )
                    else:
                        reasoning.append(
                            {
                                "stage": "Moon",
                                "rule": void_check["reason"],
                                "weight": -void_penalty,
                            }
                        )
                if getattr(config.moon, "void_gating", False):
                    return {
                        "result": "NO",
                        "confidence": max(confidence - void_penalty, 0),
                        "reasoning": reasoning,
                        "timing": None,
                    }
        
        # 2. Identify significators
        significators = self._identify_significators(chart, question_analysis)
        if not significators["valid"]:
            return {
                "result": "CANNOT JUDGE",
                "confidence": 0,
                "reasoning": reasoning + [significators["reason"]],
                "timing": None
            }
        
        desc = significators.get("description", "")
        # Fix taxonomy issue where shared significator message repeated the same house
        if desc.startswith("Shared Significator") and significators.get("same_ruler_analysis"):
            same_ruler_info = significators["same_ruler_analysis"]
            houses = list(same_ruler_info.get("houses", []))
            if len(houses) >= 2:
                # If houses are duplicated, use the individual querent/quesited houses
                if houses[0] == houses[1]:
                    q_house = significators.get("querent_house")
                    qd_house = significators.get("quesited_house")
                    if q_house and qd_house and q_house != qd_house:
                        houses = [q_house, qd_house]
                shared = same_ruler_info.get("shared_ruler")
                if shared and len(houses) >= 2:
                    desc = (
                        f"Shared Significator: {shared.value} rules both houses {houses[0]} and {houses[1]}"
                    )
                    significators["description"] = desc
                    same_ruler_info["houses"] = houses

        reasoning.append(f"Significators: {significators['description']}")
        
        querent_planet = significators["querent"] 
        quesited_planet = significators["quesited"]
        
        # CRITICAL FIX: Add pet safety scoring and Moon testimony
        pet_safety = self._calculate_pet_safety_score(chart, question_analysis, querent_planet, quesited_planet)
        total_pet_safety_score = pet_safety["score"]
        if pet_safety["score"] != 0:
            for factor_text, factor_weight in pet_safety["factors"]:
                reasoning.append({"stage": "Pet Safety", "rule": factor_text, "weight": factor_weight})
        
        # Add Moon testimony analysis
        moon_testimony = self._calculate_moon_testimony_score(chart, question_analysis, querent_planet, quesited_planet)
        total_moon_score = moon_testimony["score"]
        if moon_testimony["score"] != 0:
            for factor_text, factor_weight in moon_testimony["factors"]:
                reasoning.append({"stage": "Moon Testimony", "rule": factor_text, "weight": factor_weight})
        
        # CRITICAL FIX: Check for malefic prohibition from significators WITH CHRONOLOGY GATE
        # This applies to all question types as a general traditional principle
        relevant_house = question_analysis.get("relevant_houses", [None])[1] if len(question_analysis.get("relevant_houses", [])) > 1 else None
        
        # Get earliest perfection timing to check chronology
        earliest_perfection_days = self._get_earliest_perfection_timing(chart, querent_planet, quesited_planet)
        
        querent_prohibition = self._check_malefic_prohibition(chart, querent_planet, relevant_house, earliest_perfection_days)
        quesited_prohibition = self._check_malefic_prohibition(chart, quesited_planet, relevant_house, earliest_perfection_days)
        
        if querent_prohibition["has_prohibition"]:
            for prohibition in querent_prohibition["prohibitions"]:
                # Use proper stage based on chronology
                stage = ("Post-perfection corruption" if prohibition.get("chronology_checked") and 
                        "post-perfection complication" in prohibition["reasoning"] 
                        else "Prohibition")
                reasoning.append({
                    "stage": stage, 
                    "rule": f"L1 {prohibition['reasoning']}", 
                    "weight": prohibition["weight"]
                })
                
        if quesited_prohibition["has_prohibition"]:
            for prohibition in quesited_prohibition["prohibitions"]:
                # Use proper stage based on chronology  
                stage = ("Post-perfection corruption" if prohibition.get("chronology_checked") and 
                        "post-perfection complication" in prohibition["reasoning"]
                        else "Prohibition")
                reasoning.append({
                    "stage": stage, 
                    "rule": f"L{relevant_house or 'X'} {prohibition['reasoning']}", 
                    "weight": prohibition["weight"]
                })
        
        # Enhanced same-ruler analysis (Fix 2 & 5)
        same_ruler_bonus = 0
        if significators.get("same_ruler_analysis"):
            same_ruler_info = significators["same_ruler_analysis"]
            reasoning.append(f"Unity factor: {same_ruler_info['interpretation']}")
            
            # Traditional horary: same ruler = favorable disposition
            same_ruler_bonus = 10  # Moderate bonus for unity of purpose
            
            # However, must analyze the shared planet's condition more carefully
            shared_planet = same_ruler_info["shared_ruler"]
            shared_position = chart.planets[shared_planet]
            
            if shared_position.dignity_score > 0:
                same_ruler_bonus += 5  # Well-dignified shared ruler is very favorable
                reasoning.append(f"Shared significator {shared_planet.value} is well-dignified (+{shared_position.dignity_score})")
            elif shared_position.dignity_score < -10:
                same_ruler_bonus -= 10  # Severely debilitated shared ruler reduces unity benefit
                reasoning.append(f"Shared significator {shared_planet.value} is severely debilitated ({shared_position.dignity_score})")
            
            confidence += same_ruler_bonus
        
        # Enhanced solar condition analysis
        solar_factors = self._analyze_enhanced_solar_factors(
            chart, querent_planet, quesited_planet, ignore_combustion)
        
        if solar_factors["significant"]:
            # Don't add generic solar conditions message - will be added below with context
            solar_config = getattr(config, "solar", None)
            r17b_enabled = getattr(solar_config, "severe_impediment_denial_enabled", False) if solar_config else False

            # ENHANCED: Adjust confidence based on solar conditions affecting SIGNIFICATORS
            if solar_factors["cazimi_count"] > 0:
                confidence += config.confidence.solar.cazimi_bonus
                reasoning.append("Cazimi planets significantly strengthen the judgment")
            elif (solar_factors["combustion_count"] > 0 or solar_factors["under_beams_count"] > 0) and not ignore_combustion:
                penalty_reasons = []  # list of (reason, penalty)
                solar_penalty = 0
                severe_impediments = 0

                # Include Moon as co-significator in education charts 
                planets_to_check = [querent_planet, quesited_planet]
                if question_analysis.get("category") == "education" or getattr(chart, "category", "") == "education":
                    planets_to_check.append(Planet.MOON)
                
                for planet in planets_to_check:
                    planet_analysis = solar_factors["detailed_analyses"].get(planet.value, {})
                    condition = planet_analysis.get("condition")
                    if condition not in ["Combustion", "Under the Beams"]:
                        continue

                    distance = planet_analysis.get("distance_from_sun", 0)
                    planet_dignity = chart.planets[planet].dignity_score

                    if condition == "Combustion":
                        if distance < 1.0:
                            severe_impediments += 1
                            solar_penalty += 40
                            reason = f"{planet.value} (extreme combustion at {distance:.1f}°)"
                            penalty = 40
                        elif distance < 2.0:
                            solar_penalty += 25
                            reason = f"{planet.value} (severe combustion at {distance:.1f}°)"
                            penalty = 25
                        elif distance < 5.0:
                            solar_penalty += 15
                            reason = f"{planet.value} (combustion at {distance:.1f}°)"
                            penalty = 15
                        else:
                            solar_penalty += 10
                            reason = f"{planet.value} (light combustion at {distance:.1f}°)"
                            penalty = 10

                        if planet_dignity <= -4 and distance < 3.0:
                            severe_impediments += 1
                            reason += f" (also severely debilitated: {planet_dignity:+d})"

                        penalty_reasons.append((reason, penalty))

                    elif condition == "Under the Beams":
                        ub_penalty = config.confidence.solar.under_beams_penalty
                        solar_penalty += ub_penalty
                        penalty_reasons.append((f"{planet.value} under beams", ub_penalty))

                if r17b_enabled and severe_impediments >= 2:
                    return {
                        "result": "NO",
                        "confidence": 90,
                        "reasoning": reasoning + [f"Multiple severe solar impediments deny perfection: {', '.join(penalty_reasons)}"],
                        "timing": None,
                        "traditional_factors": {
                            "perfection_type": "none",  # FIX: Allow hybrid confidence to apply testimony adjustments
                            "impediment_type": "severe_combustion_and_debilitation"
                        },
                        "solar_factors": solar_factors,
                        "hybrid_confidence_needed": True  # FIX: Enable hybrid confidence calculation
                    }

                if penalty_reasons:
                    applied_penalty = min(solar_penalty, 50)
                    confidence -= applied_penalty

                    applied = 0
                    weighted_reasons = []
                    for reason, penalty in penalty_reasons:
                        if applied >= applied_penalty:
                            weight = 0
                        else:
                            weight = min(penalty, applied_penalty - applied)
                            applied += weight
                        weighted_reasons.append(f"{reason} (-{weight})")

                    reasoning.append(
                        f"Solar impediment: {', '.join(weighted_reasons)}"
                    )
                else:
                    reasoning.append(f"Solar conditions: {solar_factors['summary']} (significators unaffected)")
        
        # 3. Enhanced perfection check with transaction support
        # CRITICAL FIX: Handle transaction questions with natural significators
        if significators.get("transaction_type") and significators.get("item_significator"):
            # For transaction questions, check for translation involving natural significator
            item_significator = significators["item_significator"]
            item_name = significators.get("item_name", "item")
            
            # Check for translation patterns involving the item significator
            translation_result = self._check_transaction_translation(chart, querent_planet, quesited_planet, item_significator)
            if translation_result["found"]:
                result = "YES" if translation_result["favorable"] else "NO"
                confidence = min(confidence, translation_result["confidence"])
                
                # Enhanced color-coded explanation
                direction_indicator = "Favorable" if translation_result["pattern"] == "item_to_party" else "Mixed"
                reasoning.append(f"{direction_indicator} Translation Found: {translation_result['reason']}")
                
                # Explain why this indicates success/failure
                if translation_result["pattern"] == "item_to_party":
                    party = "seller" if "seller" in translation_result["reason"] else "buyer"
                    reasoning.append(f"Success Pattern: Item's energy flows to {party} - transaction completes")
                elif translation_result["pattern"] == "party_to_item":
                    party = "seller" if "seller" in translation_result["reason"] else "buyer"
                    reasoning.append(f"Mixed Pattern: {party.title()}'s energy flows to item - potential but uncertain")
                
                timing = self._calculate_enhanced_timing(chart, translation_result)
                
                return {
                    "result": result,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "timing": timing,
                    "traditional_factors": {
                        "perfection_type": "transaction_translation",
                        "reception": translation_result.get("reception", "none"),
                        "querent_strength": chart.planets[querent_planet].dignity_score,
                        "quesited_strength": chart.planets[quesited_planet].dignity_score,
                        f"{item_name}_strength": chart.planets[item_significator].dignity_score
                    },
                    "solar_factors": solar_factors
                }
        
        # Standard perfection check for non-transaction questions
        # SPECIAL HANDLING: For 3rd person education questions, check perfection between student and success
        primary_significator = querent_planet
        secondary_significator = quesited_planet
        
        if significators.get("third_person_education"):
            # For 3rd person education: analyze student -> success, not teacher -> success
            primary_significator = significators["student"]
            secondary_significator = significators["quesited"]  # Success
            reasoning.append(f"3rd person analysis: Student ({primary_significator.value}) seeking Success ({secondary_significator.value})")
        
        perfection = self._check_enhanced_perfection(
            chart, primary_significator, secondary_significator, exaltation_confidence_boost, window_days
        )

        # Post-event mode: count recent separating aspects as positive testimony
        if (
            not perfection.get("perfects")
            and question_analysis.get("post_event")
        ):
            sep = self._find_separating_aspect(
                chart, primary_significator, secondary_significator
            )
            if sep and sep.get("orb") is not None:
                orb_limit = max(sep["aspect"].orb, 8.0)
                if sep["orb"] <= orb_limit:
                    perfection = {
                        "perfects": True,
                        "type": "separating",
                        "favorable": True,
                        "confidence": cfg().confidence.perfection.direct_basic,
                        "reason": f"Recent separation: {self._format_aspect_for_display(primary_significator.value, sep['aspect'], secondary_significator.value, False)}",
                        "aspect": sep,
                    }

        # Handle explicit refranation before other checks
        if perfection.get("type") == "refranation":
            return {
                "result": "NO",
                "confidence": min(confidence, perfection.get("confidence", cfg().confidence.denial.refranation)),
                "reasoning": reasoning + [f"Refranation: {perfection['reason']}"] ,
                "timing": None,
                "traditional_factors": {
                    "perfection_type": "refranation",
                    "querent_strength": chart.planets[querent_planet].dignity_score,
                    "quesited_strength": chart.planets[quesited_planet].dignity_score,
                    "reception": self._detect_reception_between_planets(chart, primary_significator, secondary_significator),
                },
                "solar_factors": solar_factors,
            }

        # If a direct aspect exists, handle frustration or immediate denial before considering Moon aspects
        if "aspect" in perfection:
            frustration_result = self._check_frustration(
                chart, primary_significator, secondary_significator
            )
            if frustration_result.get("found"):
                return {
                    "result": "NO",
                    "confidence": min(confidence, frustration_result["confidence"]),
                    "reasoning": reasoning
                    + [f"Frustration: {frustration_result['reason']}"],
                    "timing": None,
                    "traditional_factors": {
                        "perfection_type": "frustration",
                        "frustrating_planet": frustration_result["frustrating_planet"].value,
                        "reception": frustration_result.get("reception", "none"),
                        "querent_strength": chart.planets[querent_planet].dignity_score,
                        "quesited_strength": chart.planets[quesited_planet].dignity_score,
                    },
                    "solar_factors": solar_factors,
                }

        # GENERAL ENHANCEMENT: Check Moon-Sun aspects in education questions (traditional co-significator analysis)
        if not perfection["perfects"] and question_type == Category.EDUCATION:
            moon_sun_perfection = self._check_moon_sun_education_perfection(
                chart, question_analysis
            )
            if moon_sun_perfection["perfects"]:
                perfection = moon_sun_perfection
                reasoning.append(
                    f"Moon-Sun education perfection: {moon_sun_perfection['reason']}"
                )

        # PRIORITY: Determine Moon's next aspect before applying other adjustments
        moon_next_aspect_result = self._check_moon_next_aspect_to_significators(
            chart, querent_planet, quesited_planet, ignore_void_moon
        )
        if perfection.get("perfects"):
            moon_next_aspect_result["decisive"] = False

        sun_to_10th = None
        defaults = get_defaults(question_type) if question_type else {}
        if 10 in defaults.get("houses", []):
            sun_to_10th = self._check_sun_applying_to_10th_ruler(chart)
            if sun_to_10th:
                reasoning.append(
                    "Sun applying to 10th ruler - result/recognition revealed soon"
                )
                confidence = min(100, confidence + 2)
                if sun_to_10th.get("combust"):
                    reasoning.append("Combustion on 10th ruler mitigated")

        # CRITICAL FIX: Set evaluation focus based on question intent  
        occurrence_focused = (question_intent == "OCCURRENCE")
        
        # Apply intent-based logic adjustments throughout the existing evaluation
        if occurrence_focused:
            reasoning.append("FOCUS: Evaluating occurrence - 'Will it happen?' (perfection-centered)")
        else:  # QUALITY focused
            reasoning.append("FOCUS: Evaluating quality - 'Is it favorable?' (condition-centered)")
            # For quality questions, boost confidence floor to avoid harsh NO when conditions are decent
            confidence = max(confidence, 35)

        # Initialize result based on perfection (CRITICAL FIX for UnboundLocalError)
        if perfection["perfects"]:
            # Set flag for penalty functions to respect perfection minimum
            self._has_valid_perfection = True
            
            # Initialize result based on perfection favorability
            result = "YES" if perfection.get("favorable", True) else "NO"

            # CRITICAL FIX 1: Apply separating aspect penalty
            confidence = self._apply_aspect_direction_adjustment(
                confidence, perfection, reasoning
            )

            # Incorporate Moon's next aspect testimony before other penalties
            if moon_next_aspect_result.get("result"):
                moon_confidence = moon_next_aspect_result.get("confidence", confidence)
                
                # CRITICAL FIX: Assign proper weight to primary perfections
                if moon_next_aspect_result.get("result") == "NO":
                    # Negative testimony - calculate penalty weight
                    weight = max(10, int(confidence - moon_confidence)) if moon_confidence < confidence else 10
                    # CRITICAL FIX: Respect separated confidence minimum
                    new_conf = max(separated_minimum, confidence - weight)
                    reasoning.append(
                        f"Moon's next aspect denies perfection: {moon_next_aspect_result['reason']} (-{weight})"
                    )
                else:
                    # Positive testimony - assign significant weight for primary perfection
                    if moon_next_aspect_result.get("decisive") or "trine" in moon_next_aspect_result.get("reason", "").lower():
                        weight = 15  # Major perfection weight
                    else:
                        weight = 10  # Standard supportive weight
                    
                    new_conf = min(100, confidence + weight)
                    
                    if not added_moon_support_note:
                        reasoning.append(
                            f"Moon's next aspect applies and supports; treated as supportive testimony (primary perfection handled separately): {moon_next_aspect_result['reason']} (+{weight})"
                        )
                        added_moon_support_note = True
                
                confidence = new_conf

            if moon_next_aspect_result.get("decisive"):
                reasoning.append("FLAG: MOON_NEXT_DECISIVE")

            # CRITICAL FIX 2: Apply retrograde quesited penalty early so bonuses can offset it
            confidence = self._apply_retrograde_quesited_penalty(
                confidence, chart, quesited_planet, reasoning
            )

            # CRITICAL FIX 3: Apply dignity-based confidence adjustment (can mitigate retrograde)
            confidence = self._apply_dignity_confidence_adjustment(
                confidence, chart, querent_planet, quesited_planet, reasoning
            )

            if (
                moon_next_aspect_result.get("decisive")
                and moon_next_aspect_result.get("result") == "YES"
            ):
                confidence = max(confidence, 30)

            # Clear step-by-step traditional reasoning with proper weight assignment
            if perfection["type"] == "direct_penalized":
                reasoning.append({"stage": "Perfection found", "rule": f"Direct aspect penalized: {perfection['reason']}", "weight": -5})
            elif perfection["favorable"]:
                # CRITICAL FIX: Assign contextual weight for collection of light
                if perfection["type"] == "collection":
                    collector = perfection.get("collector")
                    collector_pos = chart.planets.get(collector) if collector else None
                    
                    if collector_pos:
                        # Contextual collection scoring
                        if collector in {Planet.JUPITER, Planet.VENUS}:  # Benefic collector
                            if collector_pos.dignity_score >= 3:
                                weight = 6  # Benefic collector with reception
                            else:
                                weight = 4  # Benefic collector without strong reception
                        elif collector == Planet.SATURN:  # Saturn collection
                            if collector_pos.dignity_score >= 0:
                                weight = 2  # Saturn collection with decent dignity (help via authority)
                            else:
                                weight = -1  # Saturn collection with poor dignity (heavy obligation)
                        else:  # Other collectors (Mars, etc.)
                            weight = 1  # Neutral collection
                        
                        # House condition modifier
                        if collector_pos.house in [1, 4, 7, 10]:  # Angular houses
                            weight += 1
                        elif collector_pos.house in [6, 8, 12]:  # Cadent malefic houses
                            weight -= 1
                    else:
                        weight = 2  # Default collection weight
                else:
                    weight = 8  # Other types of perfection
                    
                # INTENT-AWARE: For quality questions, consider dignity context with perfection
                if not occurrence_focused:
                    querent_dignity = chart.planets[querent_planet].dignity_score
                    quesited_dignity = chart.planets[quesited_planet].dignity_score
                    
                    # If perfection exists but dignities are poor, reduce weight for quality assessment
                    if querent_dignity < -2 and quesited_dignity < -2:
                        weight = max(1, weight - 3)
                        reasoning.append("QUALITY FOCUS: Perfection weight reduced due to poor dignities")
                
                reasoning.append({"stage": "Perfection found", "rule": perfection['reason'], "weight": weight})
            else:
                neg_detail = ""
                if perfection.get("negative_reasons"):
                    neg_detail = f" ({'; '.join(perfection['negative_reasons'])})"
                reasoning.append({"stage": "Perfection found", "rule": f"❌ Negative perfection: {perfection['reason']}{neg_detail}", "weight": -8})

            # Apply consideration penalties (R1, R26) after perfection
            # For perfection cases, maintain minimum floor confidence
            penalty_floor = 25 if perfection.get("type") in ["translation", "collection"] else 0
            confidence = max(confidence - asc_penalty - void_penalty, penalty_floor)

            # Apply debilitated ruler and cadent significator penalties
            confidence = self._apply_debilitation_and_cadent_penalties(
                confidence,
                chart,
                querent_planet,
                quesited_planet,
                reasoning,
                category_rules,
            )

            # Apply timing decay based on perfection timing
            confidence = int(
                self._apply_timing_decay(confidence, perfection.get("t_perfect_days"))
            )

            # CRITICAL FIX 4: Apply confidence threshold (FIXED - low confidence should be NO/INCONCLUSIVE)
            result, confidence = self._apply_confidence_threshold(
                result, confidence, reasoning
            )

            # Enhanced timing with real Moon speed
            timing = self._calculate_enhanced_timing(chart, perfection)

            return {
                "result": result,
                "confidence": confidence,
                "reasoning": reasoning,
                "timing": timing,
                "traditional_factors": {
                    "perfection_type": perfection["type"],
                    "reception": perfection.get("reception", "none"),
                    "querent_strength": chart.planets[querent_planet].dignity_score,
                    "quesited_strength": chart.planets[quesited_planet].dignity_score,
                },
                "solar_factors": solar_factors,
            }
        else:
            # No perfection found - initialize default result
            result = "NO"

        # Gather potential blockers before considering special overrides
        blocker_eval = self._evaluate_blockers(
            chart,
            querent_planet,
            quesited_planet,
            moon_next_aspect_result,
            solar_factors,
            ignore_void_moon,
            ignore_combustion,
        )
        if blocker_eval["fatal"]:
            top_blocker = blocker_eval["blockers"][0]
            return {
                "result": "NO",
                "confidence": confidence,  # Preserve original confidence for hybrid calculation
                "reasoning": reasoning + [top_blocker["reason"]],
                "timing": None,
                "traditional_factors": {
                    "perfection_type": "blocked",
                    "blockers": blocker_eval["blockers"],
                    "reception": "none",
                    "querent_strength": chart.planets[querent_planet].dignity_score,
                    "quesited_strength": chart.planets[quesited_planet].dignity_score,
                },
                "solar_factors": solar_factors,
                "hybrid_confidence_needed": True,  # Flag for hybrid calculation
            }

        # 3.5. Traditional Same-Ruler Logic (FIXED: Unity defaults to YES unless explicit prohibition)
        if significators.get("same_ruler_analysis"):
            same_ruler_info = significators["same_ruler_analysis"]
            shared_planet = same_ruler_info["shared_ruler"]
            shared_position = chart.planets[shared_planet]
            
            # Unity indicates perfection - default to YES
            result = "YES"
            base_confidence = 75  # Good confidence for unity
            timing_description = "Moderate timeframe"
            
            # Check for explicit prohibitions that could deny the unity
            prohibitions = []
            
            # Check for severe debilitation that could deny
            if shared_position.dignity_score <= -10:
                prohibitions.append("Shared significator severely debilitated")
            
            # Check for combustion (if not ignored)
            if not ignore_combustion and any(
                analysis.condition in ["Combustion", "Under the Beams"] 
                for analysis in solar_factors.get("detailed_analyses", {}).values() 
                if analysis["planet"] == shared_planet
            ):
                prohibitions.append("Shared significator combust/under beams")
            
            # Check for explicit refranation or frustration
            if shared_position.retrograde and shared_position.dignity_score < -5:
                prohibitions.append("Shared significator retrograde and weak (refranation)")
            
            # If explicit prohibitions exist, deny
            if prohibitions:
                result = "NO"
                base_confidence = 80
                reasoning.append(f"Same ruler unity denied: {', '.join(prohibitions)}")
            else:
                # Unity perfected - check for conditions/modifications
                conditions = []
                
                # Retrograde indicates delays/conditions, not denial
                if shared_position.retrograde:
                    conditions.append("with delays/renegotiation (retrograde)")
                    timing_description = "Delayed/with conditions"
                
                # Poor dignity indicates difficulty but not denial
                if -10 < shared_position.dignity_score < 0:
                    conditions.append("with difficulty")
                
                if conditions:
                    result = "YES"
                    reasoning.append(f"Same ruler unity perfected {' '.join(conditions)}")
                else:
                    reasoning.append("Same ruler unity indicates direct perfection")
            
            # Get Moon testimony for confidence modification (not decisive)
            moon_testimony = self._check_enhanced_moon_testimony(chart, querent_planet, quesited_planet, ignore_void_moon)
            
            # FIXED: Detect conflicting testimonies and adjust confidence
            reception = self._detect_reception_between_planets(chart, querent_planet, quesited_planet)
            has_reception = reception != "none"
            
            # Count positive vs negative testimonies for conflict detection
            positive_testimonies = []
            negative_testimonies = []
            
            # Unity itself is positive
            positive_testimonies.append("same ruler unity")
            
            # Reception is positive
            if has_reception:
                positive_testimonies.append(f"reception ({reception})")
            
            # Analyze Moon aspects for conflicts
            if moon_testimony.get("aspects"):
                favorable_aspects = [a for a in moon_testimony["aspects"] if a.get("favorable")]
                unfavorable_aspects = [a for a in moon_testimony["aspects"] if not a.get("favorable")]
                
                if favorable_aspects:
                    positive_testimonies.extend([a["description"] for a in favorable_aspects])
                if unfavorable_aspects:
                    negative_testimonies.extend([a["description"] for a in unfavorable_aspects])
            
            # Calculate confidence based on testimony balance
            if positive_testimonies and negative_testimonies:
                # Conflicting testimonies - reduce confidence
                testimony_conflict_penalty = min(15, len(negative_testimonies) * 5)
                base_confidence = max(65, base_confidence - testimony_conflict_penalty)
                reasoning.append(f"Conflicting testimonies reduce certainty ({len(positive_testimonies)} positive, {len(negative_testimonies)} negative)")
            elif moon_testimony.get("favorable"):
                base_confidence = min(85, base_confidence + 5)
                reasoning.append(f"Moon supports unity: {moon_testimony['reason']}")
            elif moon_testimony.get("unfavorable"):
                base_confidence = max(70, base_confidence - 5)  # Reduced penalty due to unity
                reasoning.append(f"Moon testimony concerning but unity remains: {moon_testimony['reason']}")
            
            # Reception bonus
            if has_reception:
                base_confidence = min(90, base_confidence + 3)
                reasoning.append(f"Reception supports perfection: {reception}")
            
            # FIXED: Check Moon's dual roles (house ruler vs co-significator)
            moon_house_roles = []
            for house_num, ruler in chart.house_rulers.items():
                if ruler == Planet.MOON:
                    moon_house_roles.append(house_num)
            
            if moon_house_roles:
                relevant_moon_roles = []
                for house in moon_house_roles:
                    if house in [1, 2, 7, 8, 10, 11]:  # Houses potentially relevant to financial/approval questions
                        relevant_moon_roles.append(house)
                
                if relevant_moon_roles:
                    # Moon as house ruler should be analyzed separately from general testimony
                    moon_as_ruler_condition = chart.planets[Planet.MOON]
                    
                    if moon_as_ruler_condition.dignity_score >= 0:
                        base_confidence = min(88, base_confidence + 3)
                        reasoning.append(f"Moon as L{',L'.join(map(str, relevant_moon_roles))} well-positioned supports perfection")
                    elif moon_as_ruler_condition.dignity_score < -5:
                        base_confidence = max(65, base_confidence - 5)
                        reasoning.append(f"Moon as L{',L'.join(map(str, relevant_moon_roles))} poorly positioned creates uncertainty")
                    
                    # For loan applications, L10 (authority) is especially important
                    if 10 in relevant_moon_roles:
                        reasoning.append("Moon as L10 (authority/decision-maker) is key to approval process")
            
            timing = self._calculate_enhanced_timing(chart, {"type": "same_ruler_unity", "planet": shared_planet})
            
            return {
                "result": result,
                "confidence": base_confidence,
                "reasoning": reasoning,
                "timing": timing,
                "traditional_factors": {
                    "perfection_type": "same_ruler_unity",
                    "reception": self._detect_reception_between_planets(chart, querent_planet, quesited_planet),
                    "querent_strength": shared_position.dignity_score,
                    "quesited_strength": shared_position.dignity_score,  # Same ruler = same strength
                    "moon_void": moon_testimony.get("void_of_course", False)
                },
                "solar_factors": solar_factors
            }
        
        # 3.6. PRIORITY: Moon's next applying aspect to significators (traditional key indicator)
        if moon_next_aspect_result.get("result"):
            moon_confidence = moon_next_aspect_result.get("confidence", confidence)
            
            # CRITICAL FIX: Assign proper weight to primary perfections (duplicate fix)
            if moon_next_aspect_result.get("result") == "NO":
                # Negative testimony - calculate penalty weight
                weight = max(10, int(confidence - moon_confidence)) if moon_confidence < confidence else 10
                # CRITICAL FIX: Respect separated confidence minimum
                new_conf = max(separated_minimum, confidence - weight)
                reasoning.append(
                    f"Moon's next aspect denies perfection: {moon_next_aspect_result['reason']} (-{weight})"
                )
            else:
                # Positive testimony - assign significant weight for primary perfection
                if moon_next_aspect_result.get("decisive") or "trine" in moon_next_aspect_result.get("reason", "").lower():
                    weight = 15  # Major perfection weight
                else:
                    weight = 10  # Standard supportive weight
                
                new_conf = min(100, confidence + weight)
                
                if not added_moon_support_note:
                    reasoning.append(
                        f"Moon's next aspect applies and supports; treated as supportive testimony (primary perfection handled separately): {moon_next_aspect_result['reason']} (+{weight})"
                    )
                    added_moon_support_note = True
            
            confidence = new_conf

        if moon_next_aspect_result.get("decisive"):
            reasoning.append("FLAG: MOON_NEXT_DECISIVE")
        
        # 3.7. Enhanced Moon testimony analysis when no decisive Moon aspect
        moon_testimony = self._check_enhanced_moon_testimony(chart, querent_planet, quesited_planet, ignore_void_moon)
        
        # 4. Enhanced denial conditions (retrograde now configurable)
        denial = self._check_enhanced_denial_conditions(chart, querent_planet, quesited_planet)
        if denial["denied"]:
            return {
                "result": "NO",
                "confidence": min(confidence, denial["confidence"]),
                "reasoning": reasoning + [f"Denial: {denial['reason']}"],
                "timing": None,
                "solar_factors": solar_factors
            }
        
        # 4.5. ENHANCED: Check theft/loss-specific denial factors
        theft_denials = self._check_theft_loss_specific_denials(chart, question_type, querent_planet, quesited_planet)
        if theft_denials:
            combined_theft_denial = "; ".join(theft_denials)
            return {
                "result": "NO", 
                "confidence": 80,  # High confidence for traditional theft denial factors
                "reasoning": reasoning + [f"Theft/Loss Denial: {combined_theft_denial}"],
                "timing": None,
                "solar_factors": solar_factors,
                "traditional_factors": {
                    "perfection_type": "none",  # FIX: Allow hybrid confidence to apply testimony adjustments
                    "denial_type": "theft_loss_specific"
                },
                "hybrid_confidence_needed": True  # FIX: Enable hybrid confidence calculation
            }
        
        # 5. ENHANCED: Check benefic aspects to significators - BUT ONLY as secondary testimony
        # Traditional rule: Benefic support alone cannot override lack of significator perfection
        benefic_support = self._check_benefic_aspects_to_significators(chart, querent_planet, quesited_planet)
        benefic_support_overridden = False

        if benefic_support["favorable"]:
            # ROOT FIX: Add significator weakness assessment to benefic support logic
            quesited_pos = chart.planets[quesited_planet]

            # Check if quesited is severely debilitated
            if quesited_pos.dignity_score <= -4 or quesited_pos.retrograde:
                # Severely weak quesited overrides benefic support
                weakness_reasons = []
                if quesited_pos.dignity_score <= -4:
                    weakness_reasons.append(f"severely debilitated ({quesited_pos.dignity_score:+d})")
                if quesited_pos.retrograde:
                    weakness_reasons.append("retrograde")

                reasoning.append(f"Note: {benefic_support['reason']} (insufficient - quesited {', '.join(weakness_reasons)})")
                benefic_support_overridden = True
            else:
                # Traditional horary: benefic support noted but not decisive
                reasoning.append(f"Note: {benefic_support['reason']} (secondary testimony)")
        elif benefic_support["neutral"]:
            reasoning.append(f"Note: {benefic_support['reason']} (secondary testimony)")
        
        # 6. PREGNANCY-SPECIFIC: Check for Moon→benefic OR L1↔L5 reception (FIXED: don't auto-deny)
        if question_type == Category.PREGNANCY:
            # Check for L1↔L5 reception (already fixed)
            reception = self._detect_reception_between_planets(chart, querent_planet, quesited_planet)
            has_reception = reception != "none"
            
            # Check for Moon→benefic testimony (already fixed in moon testimony)  
            has_moon_benefic = False
            if moon_testimony.get("aspects"):
                for aspect_info in moon_testimony["aspects"]:
                    if (aspect_info.get("testimony_type") == "moon_to_benefic" and 
                        aspect_info.get("applying") and aspect_info.get("favorable")):
                        has_moon_benefic = True
                        break
            
            # Pregnancy exception: Don't auto-deny if reception OR moon→benefic exists
            if has_reception or has_moon_benefic:
                reception_reason = f"L1↔L5 reception ({reception})" if has_reception else ""
                moon_benefic_reason = "Moon applying to benefic" if has_moon_benefic else ""
                combined_reason = " & ".join(filter(None, [reception_reason, moon_benefic_reason]))
                
                reasoning.append(f"Pregnancy: {combined_reason}")
                
                # Calculate confidence based on quality of testimony
                pregnancy_confidence = 70  # Base for pregnancy sufficiency
                if has_reception:
                    pregnancy_confidence += 5
                if has_moon_benefic:
                    pregnancy_confidence += 5
                
                return {
                    "result": "YES",
                    "confidence": pregnancy_confidence,
                    "reasoning": reasoning,
                    "timing": moon_testimony.get("timing", "Moderate timeframe"),
                    "traditional_factors": {
                        "perfection_type": "pregnancy_sufficiency",
                        "reception": reception,
                        "querent_strength": chart.planets[querent_planet].dignity_score,
                        "quesited_strength": chart.planets[quesited_planet].dignity_score,
                        "moon_benefic": has_moon_benefic
                    },
                    "solar_factors": solar_factors
                }
        
        # 7. FALLBACK: Build transparent math-based denial reasoning 
        config = cfg()
        
        # CRITICAL FIX: For SAFETY mode questions, evaluate condition instead of requiring perfection
        question_intent = question_analysis.get("question_intent", "REUNION")
        if question_intent == "SAFETY":
            # Safety questions evaluate the quesited's condition, not perfection between significators
            
            # Calculate total pet safety + moon testimony score
            total_safety_score = total_pet_safety_score + total_moon_score
            
            if total_safety_score > 9:  # Strong positive testimony
                return {
                    "result": "YES",
                    "confidence": min(80 + total_safety_score, 95),
                    "reasoning": reasoning,
                    "timing": None,
                    "traditional_factors": {
                        "perfection_type": "safety_evaluation",
                        "safety_score": total_safety_score,
                        "querent_strength": chart.planets[querent_planet].dignity_score,
                        "quesited_strength": chart.planets[quesited_planet].dignity_score,
                    },
                    "solar_factors": solar_factors,
                    "hybrid_confidence_needed": True  # Still apply hybrid confidence with positive verdict
                }
            elif total_safety_score > 0:  # Moderate positive testimony
                return {
                    "result": "YES", 
                    "confidence": min(60 + total_safety_score, 80),
                    "reasoning": reasoning,
                    "timing": None,
                    "traditional_factors": {
                        "perfection_type": "safety_evaluation",
                        "safety_score": total_safety_score,
                        "querent_strength": chart.planets[querent_planet].dignity_score,
                        "quesited_strength": chart.planets[quesited_planet].dignity_score,
                    },
                    "solar_factors": solar_factors,
                    "hybrid_confidence_needed": True
                }
            # If total_safety_score <= 0, fall through to traditional denial logic
        
        # INTENT-AWARE EVALUATION: Different approach for occurrence vs quality questions
        if occurrence_focused:
            # OCCURRENCE FOCUS: Without perfection, be more strict (traditional approach)
            reasoning.append("OCCURRENCE: No perfection found - strict evaluation for event likelihood")
        else:  # QUALITY focused
            # QUALITY FOCUS: Without perfection, assess conditions more favorably
            # Quality questions care about dignity and reception even without perfection
            querent_dignity = chart.planets[querent_planet].dignity_score
            quesited_dignity = chart.planets[quesited_planet].dignity_score
            
            if querent_dignity >= 0 and quesited_dignity >= 0:
                reasoning.append("QUALITY: Decent dignities provide favorable conditions despite no perfection")
                confidence = max(confidence, 45)  # Boost confidence floor for quality assessment
        
        # Traditional REUNION mode logic - collect positive supporting signals with their values
        supportive_signals = []
        
        # Reception support
        reception_info = self.reception_calculator.calculate_comprehensive_reception(
            chart, querent_planet, quesited_planet
        )
        mutual = reception_info.get("mutual", "none")
        one_way = reception_info.get("one_way", [])
        if mutual != "none":
            reception_bonus = getattr(
                config.confidence.reception, f"{mutual}_bonus", 5
            )
            supportive_signals.append(f"{mutual} (+{reception_bonus})")
        elif one_way:
            reception_bonus = getattr(
                config.confidence.reception, "one_way_bonus", 3
            )
            supportive_signals.append(f"one-way reception (+{reception_bonus})")
        
        # Moon testimony support (check for Moon aspects to significators or benefics)
        moon_next_aspect = chart.moon_next_aspect
        if moon_next_aspect and moon_next_aspect.applying:
            # Check if Moon aspects significator or in education context, L10 ruler
            target = moon_next_aspect.planet
            if target in [querent_planet, quesited_planet]:
                if moon_next_aspect.aspect in [Aspect.TRINE, Aspect.SEXTILE]:
                    moon_bonus = 8  # Standard Moon aspect bonus
                    moon_display = f"Moon {moon_next_aspect.aspect.display_name} {target.value} (applying)"
                    supportive_signals.append(f"{moon_display} (+{moon_bonus})")
        
        # Benefic aspect support (only positive scores are truly supportive)
        benefic_score = benefic_support.get("total_score", 0)
        if benefic_score > 0:
            # Re-evaluate using enhanced scoring to ensure it's actually positive
            strongest = benefic_support.get("strongest_aspect")
            if strongest:
                # Check if this aspect should actually be negative under enhanced scoring
                aspect_desc = strongest.get("description", "")
                if "☍" in aspect_desc or "□" in aspect_desc:  # Opposition or square
                    # These should be negative under enhanced policy, don't include as supportive
                    pass
                else:
                    benefic_display = strongest.get("description", f"benefic aspect (score: {benefic_score})")
                    supportive_signals.append(f"{benefic_display} (+{benefic_score})")
            else:
                supportive_signals.append(f"benefic aspect (+{benefic_score})")

        signals_text = " and ".join(supportive_signals) if supportive_signals else "none"
        # Make explicit which significator pair was assessed for perfection
        reasoning.append(
            f"Denial: no direct perfection found between {querent_planet.value} and {quesited_planet.value}. Supportive signals noted — {signals_text}"
        )

        final_confidence = min(confidence, 75)
        if moon_next_aspect_result.get("result") == "YES" or (
            benefic_support.get("favorable") and not benefic_support_overridden
        ):
            final_confidence = min(final_confidence, 60)

        # Apply debilitated ruler and cadent penalties to final confidence
        final_confidence = self._apply_debilitation_and_cadent_penalties(
            final_confidence,
            chart,
            querent_planet,
            quesited_planet,
            reasoning,
            category_rules,
        )

        return {
            "result": "NO",
            "confidence": int(final_confidence),
            "reasoning": reasoning,
            "timing": None,
            "traditional_factors": {
                "perfection_type": "none",
                "querent_strength": chart.planets[querent_planet].dignity_score,
                "quesited_strength": chart.planets[quesited_planet].dignity_score,
                "reception": self._detect_reception_between_planets(chart, querent_planet, quesited_planet),
                "benefic_noted": benefic_support.get("total_score", 0) > 0
            },
            "solar_factors": solar_factors,
            "evidence_ledger": evidence_ledger,
            "hybrid_confidence_needed": True  # Flag for hybrid calculation
        }
    
    
    def _check_enhanced_denial_conditions(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Enhanced denial conditions with configurable retrograde handling"""
        
        config = cfg()
        
        # Traditional Frustration - any planet can aspect a significator first
        frustration_result = self._check_frustration(chart, querent, quesited)
        if frustration_result["found"]:
            return {
                "denied": True,
                "confidence": frustration_result["confidence"],
                "reason": frustration_result["reason"]
            }
        
        # Enhanced retrograde handling - configurable instead of automatic denial
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]
        reception_info = self.reception_calculator.calculate_comprehensive_reception(
            chart, querent, quesited
        )
        mutual = reception_info.get("mutual", "none")
        one_way = reception_info.get("one_way", [])
        
        if not config.retrograde.automatic_denial:
            # Retrograde is now just a penalty, not automatic denial
            if querent_pos.retrograde or quesited_pos.retrograde:
                # This will be handled in dignity scoring instead
                pass
        else:
            # Legacy behavior - automatic denial
            if querent_pos.retrograde or quesited_pos.retrograde:
                return {
                    "denied": True,
                    "confidence": config.confidence.denial.frustration_retrograde,
                    "reason": f"Frustration - {'querent' if querent_pos.retrograde else 'quesited'} significator retrograde"
                }
        
        return {"denied": False}

    def _check_enhanced_denial_conditions(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Enhanced denial conditions with configurable retrograde handling"""
        
        config = cfg()
        
        # Traditional Frustration - any planet can aspect a significator first
        frustration_result = self._check_frustration(chart, querent, quesited)
        if frustration_result["found"]:
            return {
                "denied": True,
                "confidence": frustration_result["confidence"],
                "reason": frustration_result["reason"]
            }
        
        # Enhanced retrograde handling - configurable instead of automatic denial
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]
        
        if not config.retrograde.automatic_denial:
            # Retrograde is now just a penalty, not automatic denial
            if querent_pos.retrograde or quesited_pos.retrograde:
                # This will be handled in dignity scoring instead
                pass
        else:
            # Legacy behavior - automatic denial
            if querent_pos.retrograde or quesited_pos.retrograde:
                return {
                    "denied": True,
                    "confidence": config.confidence.denial.frustration_retrograde,
                    "reason": f"Frustration - {'querent' if querent_pos.retrograde else 'quesited'} significator retrograde"
                }
        
        # ENHANCED: Travel-specific denial conditions (catch cases like EX-010)
        # Check if this is a travel question by examining chart structure
        quesited_pos = chart.planets[quesited]
        querent_pos = chart.planets[querent]
        
        # If Jupiter is the travel significator (9th house ruler) and heavily afflicted
        if quesited == Planet.JUPITER:
            travel_warnings = []
            
            # Critical: Jupiter retrograde for travel
            if quesited_pos.retrograde and quesited_pos.dignity_score < 0:
                travel_warnings.append("Jupiter (travel ruler) retrograde and debilitated")
            
            # Jupiter in 6th house (illness during travel)  
            if quesited_pos.house == 6:
                travel_warnings.append("Jupiter (travel ruler) in 6th house of illness")
            
            # Querent (Mars) in 8th house - danger, trouble
            if querent_pos.house == 8:
                travel_warnings.append("Querent in 8th house (danger/trouble)")
            
            # Moon in 6th house - health problems
            moon_pos = chart.planets[Planet.MOON]
            if moon_pos.house == 6:
                travel_warnings.append("Moon in 6th house (health concerns)")
            
            # If multiple serious travel warnings, deny
            if len(travel_warnings) >= 2:
                return {
                    "denied": True,
                    "confidence": 85,
                    "reason": f"Travel impediments: {'; '.join(travel_warnings)}"
                }
        
        return {"denied": False}
    
    def _apply_aspect_direction_adjustment(self, confidence: float, perfection: Dict, reasoning: List[str]) -> float:
        """CRITICAL FIX 1: Adjust confidence based on applying vs separating aspects"""
        
        # Check if perfection involves separating aspects
        if perfection["type"] == "direct" and "aspect" in perfection:
            aspect_info = perfection["aspect"]
            if hasattr(aspect_info, 'applying') and not aspect_info.applying:
                # Separating aspect = past opportunity, reduce confidence significantly
                penalty = 30
                confidence = max(confidence - penalty, 15)  # Minimum 15% for separating
                reasoning.append(f"Separating aspect penalty: -{penalty}% (past opportunity)")
                
        elif perfection["type"] == "translation":
            # Check if translation involves separating aspects from significators
            translator_info = perfection.get("translator_analysis", {})
            if translator_info.get("has_separating_from_significator"):
                penalty = 25
                confidence = max(confidence - penalty, 20)
                reasoning.append(f"Translation with separating component: -{penalty}%")
                
        return confidence
    
    def _apply_dignity_confidence_adjustment(self, confidence: float, chart: HoraryChart, 
                                          querent: Planet, quesited: Planet, reasoning: List[str]) -> float:
        """CRITICAL FIX 2: Adjust confidence based on significator dignities"""
        
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]
        querent_dignity = querent_pos.dignity_score
        quesited_dignity = quesited_pos.dignity_score
        
        # CRITICAL FIX: One-pass dignity calculation to avoid double-counting solar impediments
        # Check if solar impediment penalties have already been applied in reasoning
        solar_impediments_applied = any("combustion" in entry.get("rule", "").lower() or 
                                       "under the beams" in entry.get("rule", "").lower()
                                       for entry in reasoning if isinstance(entry, dict))
        
        # Quesited dignity is most critical for success
        # Only apply dignity penalties if they're not primarily due to solar impediments already counted
        if quesited_dignity <= -10:
            if not solar_impediments_applied:
                # Severely debilitated quesited = very unlikely success
                penalty = 35
                min_floor = 25 if hasattr(self, '_has_valid_perfection') and self._has_valid_perfection else 10
                confidence = max(confidence - penalty, min_floor)
                reasoning.append({"stage": "Weak quesited dignity", "rule": f"Severely weak quesited ({quesited_dignity})", "weight": -penalty})
            else:
                # Solar impediment already counted, apply reduced additional penalty for other debilities
                additional_penalty = max(0, min(10, abs(quesited_dignity) - 5))  # Reduced penalty for non-solar debilities
                if additional_penalty > 0:
                    confidence = max(confidence - additional_penalty, 25)
                    reasoning.append({"stage": "Weak quesited dignity", "rule": f"Additional debilities beyond solar impediment ({quesited_dignity})", "weight": -additional_penalty})
        elif quesited_dignity < -5:
            if not solar_impediments_applied:
                # Moderately debilitated quesited
                penalty = 20
                confidence = max(confidence - penalty, 25)
                reasoning.append({"stage": "Weak quesited dignity", "rule": f"Weak quesited dignity ({quesited_dignity})", "weight": -penalty})
            else:
                # Solar impediment already counted, minimal additional penalty
                additional_penalty = 5
                confidence = max(confidence - additional_penalty, 25)
                reasoning.append({"stage": "Weak quesited dignity", "rule": f"Minor additional debility ({quesited_dignity})", "weight": -additional_penalty})
        elif quesited_dignity >= 10:
            # Very strong quesited
            bonus = 15
            confidence = min(confidence + bonus, 95)
            reasoning.append({"stage": "Strong quesited dignity", "rule": f"Strong quesited dignity ({quesited_dignity})", "weight": bonus})
            
        # Querent dignity affects confidence but less critically
        if querent_dignity <= -10:
            penalty = 15
            confidence = max(confidence - penalty, 5)
            reasoning.append({"stage": "Weak querent dignity", "rule": f"Weak querent dignity ({querent_dignity})", "weight": -penalty})
        elif querent_dignity >= 10:
            bonus = 10
            confidence = min(confidence + bonus, 95)
            reasoning.append({"stage": "Strong querent dignity", "rule": f"Strong querent dignity ({querent_dignity})", "weight": bonus})
            
        # Minor dignity bonuses for triplicity/term/face (both significators)
        dignity_bonus_map = {"triplicity": 3, "term": 2, "face": 1}
        for planet, pos in ((querent, querent_pos), (quesited, quesited_pos)):
            for dignity in pos.dignities:
                if dignity in dignity_bonus_map:
                    bonus = dignity_bonus_map[dignity]
                    confidence = min(confidence + bonus, 95)
                    reasoning.append(f"{planet.value} has {dignity} dignity (+{bonus}%)")

        return confidence
    
    def _apply_retrograde_quesited_penalty(self, confidence: float, chart: HoraryChart,
                                         quesited: Planet, reasoning: List[str]) -> float:
        """CRITICAL FIX 2: Apply penalty for retrograde quesited"""

        config = cfg()
        quesited_pos = chart.planets[quesited]
        if quesited_pos.retrograde:
            # Retrograde quesited = turning away, obstacles, delays
            penalty = getattr(config.retrograde, "quesited_penalty", 12)
            confidence = max(confidence - penalty, 10)
            reasoning.append(f"Retrograde quesited: -{penalty}% (turning away from success)")

        return confidence

    def _apply_debilitation_and_cadent_penalties(
        self,
        confidence: float,
        chart: HoraryChart,
        querent: Planet,
        quesited: Planet,
        reasoning: List[str],
        category_rules: Dict[str, Any],
    ) -> float:
        """Apply category-aware penalties for debilitated rulers and cadent significators."""

        penalties = getattr(cfg(), "debilitation_penalties", None)
        if not penalties:
            return confidence

        threshold = getattr(penalties, "dignity_threshold", -5)

        # Only apply debilitated ruler penalties for configured outcome houses
        outcome_houses = category_rules.get("outcome_houses") or [4]
        for house in outcome_houses:
            ruler = chart.house_rulers.get(house)
            if not ruler:
                continue
            pos = chart.planets[ruler]
            if pos.dignity_score <= threshold:
                penalty = getattr(penalties, f"l{house}", 0)
                if penalty:
                    # Respect minimum confidence floor for valid perfection cases
                    min_floor = 25 if hasattr(self, '_has_valid_perfection') and self._has_valid_perfection else 0
                    confidence = max(confidence - penalty, min_floor)
                    reasoning.append(
                        f"Debilitated L{house} ruler ({ruler.value}) (-{penalty}%)"
                    )

        # Cadent significator penalty is applied only when enabled
        if "cadent_significator" in category_rules.get("scored_factors", []):
            cadent_penalty = getattr(penalties, "cadent_significator", 0)
            for planet in [querent, quesited]:
                pos = chart.planets.get(planet)
                if pos:
                    angularity = self.calculator._get_traditional_angularity(
                        pos.longitude, chart.houses, pos.house
                    )
                    if angularity == "cadent":
                        # Respect minimum confidence floor for valid perfection cases
                        min_floor = 25 if hasattr(self, '_has_valid_perfection') and self._has_valid_perfection else 0
                        confidence = max(confidence - cadent_penalty, min_floor)
                        reasoning.append(
                            f"{planet.value} in cadent house (-{cadent_penalty}%)"
                        )

        return confidence

    def _apply_timing_decay(self, confidence: float, t_perfect_days: Optional[float]) -> float:
        """Apply confidence decay for long perfection timeframes."""

        decay_cfg = getattr(getattr(cfg(), "timing", {}), "decay", None)
        if not decay_cfg or t_perfect_days is None:
            return confidence

        if t_perfect_days > getattr(decay_cfg, "long_threshold_days", float("inf")):
            return confidence * getattr(decay_cfg, "long_factor", 1.0)
        if t_perfect_days > getattr(decay_cfg, "medium_threshold_days", float("inf")):
            return confidence * getattr(decay_cfg, "medium_factor", 1.0)
        return confidence
    
    def _check_enhanced_translation_of_light(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Traditional translation of light with comprehensive validation requirements"""
        
        config = cfg()
        translation_cfg = getattr(config, "translation", SimpleNamespace())

        # Check all planets as potential translators (traditionally Moon, but allow all)
        for planet, pos in chart.planets.items():
            if planet in [querent, quesited]:
                continue
            
            
            # TRADITIONAL REQUIREMENT 1: Translator speed validation
            querent_pos = chart.planets[querent]
            quesited_pos = chart.planets[quesited]
            
            # Enhanced speed requirement - translator must be faster than both significators
            if getattr(translation_cfg, "require_speed_advantage", False):
                translator_speed = abs(pos.speed)
                querent_speed = abs(querent_pos.speed)
                quesited_speed = abs(quesited_pos.speed)
                
                # Strict speed validation: translator must be faster than both
                if not (translator_speed > querent_speed and translator_speed > quesited_speed):
                    continue
            
            # TRADITIONAL REQUIREMENT 2: Find valid aspects between translator and significators with orb validation
            # ENHANCED: Check both existing chart aspects AND future cross-sign aspects
            querent_aspect = None
            quesited_aspect = None
            
            # First, check existing aspects in chart
            for aspect in chart.aspects:
                # Check orb limits using moiety-based calculation
                if not self._is_aspect_within_orb_limits(chart, aspect):
                    continue  # Skip aspects that exceed proper orb limits
                
                if ((aspect.planet1 == planet and aspect.planet2 == querent) or
                    (aspect.planet1 == querent and aspect.planet2 == planet)):
                    querent_aspect = aspect
                elif ((aspect.planet1 == planet and aspect.planet2 == quesited) or
                      (aspect.planet1 == quesited and aspect.planet2 == planet)):
                    quesited_aspect = aspect
            
            # CRITICAL FIX: For Moon translation, always check for enhanced sequences
            # even if some aspects exist (e.g., Moon square Sun shouldn't block Moon translating to Saturn)
            if planet == Planet.MOON:
                # Check for future applying aspects after sign changes
                cross_sign_aspects = self._calculate_future_lunar_aspects(chart, querent, quesited)
                if not querent_aspect and querent in cross_sign_aspects:
                    querent_aspect = cross_sign_aspects[querent]
                if not quesited_aspect and quesited in cross_sign_aspects:
                    quesited_aspect = cross_sign_aspects[quesited]
                    
                # ENHANCED: Also check for recent separating aspects that may not be in chart.aspects
                if not querent_aspect or not quesited_aspect:
                    recent_separating = self._calculate_recent_lunar_separations(chart, querent, quesited)
                    print(f"DEBUG: Recent separating aspects found: {len(recent_separating)}")
                    for planet_key, aspect in recent_separating.items():
                        print(f"  {planet_key.value}: {aspect.aspect.display_name} {aspect.time_to_perfection:.2f} days ago")
                    
                    if not querent_aspect and querent in recent_separating:
                        querent_aspect = recent_separating[querent]
                        print(f"DEBUG: Using recent separating {querent.value} aspect: {querent_aspect.aspect.display_name}")
                    if not quesited_aspect and quesited in recent_separating:
                        quesited_aspect = recent_separating[quesited]
                        print(f"DEBUG: Using recent separating {quesited.value} aspect: {quesited_aspect.aspect.display_name}")
            
            # TRADITIONAL REQUIREMENT 2: Must have aspects to both significators
            if not (querent_aspect and quesited_aspect):
                continue
            
            # TRADITIONAL REQUIREMENT 3: Proper sequence - separate from one, apply to other with timing validation
            valid_translation = False
            sequence = ""
            separating_aspect = None
            applying_aspect = None
            
            # Case 1: Separating from querent, applying to quesited
            if (not querent_aspect.applying and quesited_aspect.applying):
                # Validate sequence timing: separation must have occurred before application
                if self._validate_translation_sequence_timing(chart, planet, querent_aspect, quesited_aspect):
                    valid_translation = True
                    sequence = f"Separates from {querent.value}, applies to {quesited.value}"
                    separating_aspect = querent_aspect
                    applying_aspect = quesited_aspect
            
            # Case 2: Separating from quesited, applying to querent  
            elif (not quesited_aspect.applying and querent_aspect.applying):
                # Validate sequence timing: separation must have occurred before application
                if self._validate_translation_sequence_timing(chart, planet, quesited_aspect, querent_aspect):
                    valid_translation = True
                    sequence = f"Separates from {quesited.value}, applies to {querent.value}"
                    separating_aspect = quesited_aspect
                    applying_aspect = querent_aspect
            
            if not valid_translation:
                continue
            
            # SUCCESS: Found valid translation
            
            # ENHANCED REQUIREMENT 4: Check for IMMEDIATE SEQUENCE (no intervening aspects)
            sequence_note = ""
            if getattr(translation_cfg, "require_proper_sequence", False):
                intervening_aspects = self._check_intervening_aspects(chart, planet, separating_aspect, applying_aspect)
                if intervening_aspects:
                    continue  # Translation invalid if other aspects intervene
                else:
                    sequence_note = " (immediate sequence)"
            
            # TRADITIONAL REQUIREMENT 5: Check reception with translator using centralized calculator
            reception_querent_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet, querent)
            reception_quesited_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet, quesited)
            
            reception_with_querent = reception_querent_data["type"] != "none"
            reception_with_quesited = reception_quesited_data["type"] != "none"
            
            # Reception helps but is not absolutely required for translation
            reception_bonus = 0
            reception_note = ""
            receptions: List[str] = []
            reception_data: List[Dict[str, Any]] = []

            if reception_with_querent:
                receptions.append(reception_querent_data["display_text"])
                reception_data.append(reception_querent_data)
            if reception_with_quesited:
                receptions.append(reception_quesited_data["display_text"])
                reception_data.append(reception_quesited_data)

            if receptions:
                reception_bonus = 10
                reception_note = " with reception"
                
            # Base confidence from traditional sources
            confidence = 65 + reception_bonus
            
            # Traditional rule: Even combust planets can translate light
            # but with reduced effectiveness
            combustion_penalty = 0
            if hasattr(pos, 'solar_condition') and pos.solar_condition.condition == "Combustion":
                combustion_penalty = 15
                confidence -= combustion_penalty
                
            # Assess favorability dynamically based on translator condition and aspects
            negative_reasons: List[str] = []
            hard = {Aspect.SQUARE, Aspect.OPPOSITION}
            if (querent_aspect.aspect in hard) or (quesited_aspect.aspect in hard):
                confidence -= 15
                negative_reasons.append("hard aspect")
            if combustion_penalty:
                confidence -= 10
                negative_reasons.append("translator combust")

            favorable = len(negative_reasons) == 0
            
            # Calculate validation metrics for transparency
            translator_speed = abs(pos.speed)
            querent_speed = abs(querent_pos.speed)
            quesited_speed = abs(quesited_pos.speed)
            
            separating_orb = separating_aspect.orb
            applying_orb = applying_aspect.orb
            separating_planet_name = querent.value if separating_aspect == querent_aspect else quesited.value
            applying_planet_name = quesited.value if applying_aspect == quesited_aspect else querent.value
            
            return {
                "found": True,
                "translator": planet,
                "favorable": favorable,
                "verdict": "favorable" if favorable else "unfavorable",
                "confidence": min(95, max(35, confidence)),  # Cap between 35-95%
                "sequence": sequence + reception_note + sequence_note,
                "receptions": receptions if receptions else ["none"],
                "reception_data": reception_data,
                "combustion_penalty": combustion_penalty,
                "challenge_reasons": [],
                "negative_reasons": negative_reasons,
                "validation_details": {
                    "speed_validated": translator_speed > max(querent_speed, quesited_speed),
                    "translator_speed": translator_speed,
                    "significator_speeds": {"querent": querent_speed, "quesited": quesited_speed},
                    "orb_validation": {
                        "separating_orb": separating_orb,
                        "applying_orb": applying_orb,
                        "separating_planet": separating_planet_name,
                        "applying_planet": applying_planet_name,
                        "orbs_within_limits": True  # We already validated this above
                    },
                    "sequence_validated": True,  # We already validated timing above
                    "intervening_aspects_checked": getattr(translation_cfg, "require_proper_sequence", False)
                }
            }
        
        return {"found": False}
    
    def _calculate_future_lunar_aspects(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[Planet, object]:
        """Calculate future lunar aspects after sign changes for translation of light detection.
        
        This detects the classic cross-sign lunar sequences that traditional aspect 
        calculations miss due to sign-boundary restrictions.
        """
        from .aspects import Aspect, _signed_longitude_delta
        import math
        
        # Create a simple aspect-like class that mimics AspectInfo interface
        class FutureAspect:
            def __init__(self, planet1, planet2, aspect_type, applying, degrees_to_exact, time_to_perfection):
                self.planet1 = planet1
                self.planet2 = planet2
                self.aspect = aspect_type
                self.applying = applying
                self.degrees_to_exact = degrees_to_exact
                self.time_to_perfection = time_to_perfection
                self.perfection_within_sign = False  # Cross-sign by definition
                self.orb = degrees_to_exact
        
        moon_pos = chart.planets[Planet.MOON]
        future_aspects = {}
        
        # Only check for Moon (fastest-moving planet for translation)
        if moon_pos.speed <= 0:
            return future_aspects
            
        # Check future aspects with each significator
        for target_planet in [querent, quesited]:
            target_pos = chart.planets[target_planet]
            
            # Look ahead up to 15 days for lunar translation sequences
            best_aspect = None
            best_time = float('inf')
            
            for days_ahead in [d * 0.5 for d in range(1, 31)]:  # Check every 0.5 days up to 15 days
                # Calculate future positions
                future_moon_lon = (moon_pos.longitude + moon_pos.speed * days_ahead) % 360
                future_target_lon = (target_pos.longitude + target_pos.speed * days_ahead) % 360
                
                # Calculate angular separation
                separation = abs(_signed_longitude_delta(future_moon_lon, future_target_lon))
                
                # Check each traditional aspect
                for aspect_type in Aspect:
                    orb_diff = abs(separation - aspect_type.degrees)
                    
                    # Use wider orb for future cross-sign aspects
                    max_orb = aspect_type.orb * 1.5
                    
                    if orb_diff <= max_orb:
                        # Check if Moon will have crossed at least one sign boundary
                        current_sign = int(moon_pos.longitude // 30)
                        future_sign = int(future_moon_lon // 30) 
                        
                        if future_sign != current_sign:  # Cross-sign aspect
                            # Create future aspect with proper interface
                            future_aspect = FutureAspect(
                                planet1=Planet.MOON,
                                planet2=target_planet,
                                aspect_type=aspect_type,
                                applying=True,  # Future aspects are applying by definition
                                degrees_to_exact=orb_diff,
                                time_to_perfection=days_ahead
                            )
                            
                            # Keep the earliest (best) aspect found
                            if days_ahead < best_time:
                                best_aspect = future_aspect
                                best_time = days_ahead
                                
            if best_aspect:
                future_aspects[target_planet] = best_aspect
                            
        return future_aspects

    def _calculate_future_planet_aspects(
        self,
        chart: HoraryChart,
        translator: Planet,
        targets: List[Planet],
        max_days: float,
    ) -> Dict[Planet, object]:
        """Generic future aspect calculator for non-lunar translators.

        Uses the analytical ``_calculate_future_aspect_time`` helper to find the
        earliest applying aspect between ``translator`` and each planet in
        ``targets``. Cross-sign aspects are naturally included since no sign
        boundary restrictions are imposed. A simple AspectInfo-like object is
        returned for each target planet containing ``time_to_perfection`` and the
        ``aspect`` type that perfects first.
        """

        from .aspects import Aspect

        class FutureAspect:
            def __init__(self, planet1, planet2, aspect_type, time_to_perfection):
                self.planet1 = planet1
                self.planet2 = planet2
                self.aspect = aspect_type
                self.applying = True
                self.degrees_to_exact = 0.0
                self.time_to_perfection = time_to_perfection
                self.perfection_within_sign = False
                self.orb = 0.0

        future_aspects: Dict[Planet, FutureAspect] = {}
        translator_pos = chart.planets[translator]

        for target in targets:
            target_pos = chart.planets.get(target)
            if not target_pos:
                continue

            best_t = float("inf")
            best_aspect: Optional[FutureAspect] = None

            for aspect_type in Aspect:
                t = self._calculate_future_aspect_time(
                    translator_pos, target_pos, aspect_type, chart.julian_day, max_days
                )
                if t is not None and t < best_t:
                    best_t = t
                    best_aspect = FutureAspect(translator, target, aspect_type, t)

            if best_aspect:
                future_aspects[target] = best_aspect

        return future_aspects
    
    def _calculate_recent_lunar_separations(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[Planet, object]:
        """Calculate recent lunar separating aspects using pure time-based analysis.
        
        CRITICAL FIX: Does not rely on current orbs - only time to perfection.
        This catches aspects that were exact recently but are now way outside normal orbs.
        """
        from .aspects import Aspect, time_to_perfection
        import math
        
        # Create aspect-like class for separating aspects
        class RecentSeparatingAspect:
            def __init__(self, planet1, planet2, aspect_type, applying, degrees_to_exact, time_to_perfection):
                self.planet1 = planet1
                self.planet2 = planet2
                self.aspect = aspect_type
                self.applying = applying  # Will be False for separating
                self.degrees_to_exact = degrees_to_exact
                self.time_to_perfection = time_to_perfection
                self.perfection_within_sign = True  # Recent aspects usually within same sign
                self.orb = degrees_to_exact
        
        moon_pos = chart.planets[Planet.MOON]
        separating_aspects = {}
        
        # Only check for Moon (fastest-moving planet for translation)
        if moon_pos.speed <= 0:
            return separating_aspects
            
        # Check recent separating aspects with each significator
        for target_planet in [querent, quesited]:
            target_pos = chart.planets[target_planet]
            best_separating = None
            most_recent_time = float('inf')
            
            # Check each traditional aspect for recent exact separations
            for aspect_type in Aspect:
                # Calculate time to perfection (negative means past perfection = separating)
                t = time_to_perfection(moon_pos, target_pos, aspect_type)
                
                # PURE TIME-BASED CHECK: If aspect was exact within last 7 days, include it
                # regardless of current orb (Moon may have moved 100+ degrees since exactness)
                if t < 0 and abs(t) <= 7:  # Separated within last 7 days
                    time_since_exact = abs(t)
                    
                    # Keep the most recent separating aspect for this planet
                    if time_since_exact < most_recent_time:
                        most_recent_time = time_since_exact
                        
                        # Distance Moon has traveled since exactness
                        degrees_traveled = time_since_exact * moon_pos.speed
                        
                        best_separating = RecentSeparatingAspect(
                            planet1=Planet.MOON,
                            planet2=target_planet,
                            aspect_type=aspect_type,
                            applying=False,  # Separating by definition
                            degrees_to_exact=degrees_traveled,  # How far we've moved since exactness
                            time_to_perfection=t  # Negative value (days since perfection)
                        )
            
            if best_separating:
                separating_aspects[target_planet] = best_separating
                            
        return separating_aspects
    
    def _check_transaction_translation(self, chart: HoraryChart, seller: Planet, buyer: Planet, item: Planet) -> Dict[str, Any]:
        """Check for translation involving transaction (seller, buyer, item) - matches reference analysis"""
        
        # Reference: Mercury translated light between Mars (buyer) and Sun (car)
        # Our version: Check if any planet translates between seller/buyer and item
        
        for translator_planet, pos in chart.planets.items():
            if translator_planet in [seller, buyer, item]:
                continue
            
            # Find aspects involving the translator
            translator_aspects = []
            for aspect in chart.aspects:
                if aspect.planet1 == translator_planet or aspect.planet2 == translator_planet:
                    other_planet = aspect.planet2 if aspect.planet1 == translator_planet else aspect.planet1
                    translator_aspects.append({
                        "other": other_planet,
                        "aspect": aspect,
                        "applying": aspect.applying,
                        "degrees_to_exact": aspect.degrees_to_exact
                    })
            
            # Check for transaction translation patterns:
            # Pattern 1: Translator separates from item, applies to seller/buyer
            # Pattern 2: Translator separates from seller/buyer, applies to item
            item_aspects = [a for a in translator_aspects if a["other"] == item]
            seller_aspects = [a for a in translator_aspects if a["other"] == seller]
            buyer_aspects = [a for a in translator_aspects if a["other"] == buyer]
            
            # Check various translation patterns
            for item_aspect in item_aspects:
                for party_aspect in seller_aspects + buyer_aspects:
                    # Translation pattern: separating from one, applying to other
                    if (not item_aspect["applying"] and party_aspect["applying"]):
                        confidence = 75
                        
                        # Reduce confidence if translator is combust
                        if hasattr(pos, 'solar_condition') and pos.solar_condition.condition == "Combustion":
                            confidence -= 10
                        
                        party_name = "seller" if party_aspect["other"] == seller else "buyer"
                        return {
                            "found": True,
                            "favorable": True,
                            "confidence": confidence,
                            "reason": f"{translator_planet.value} translates light from {item.value} (item) to {party_aspect['other'].value} ({party_name})",
                            "translator": translator_planet,
                            "pattern": "item_to_party"
                        }
                    elif (not party_aspect["applying"] and item_aspect["applying"]):
                        confidence = 75
                        
                        if hasattr(pos, 'solar_condition') and pos.solar_condition.condition == "Combustion":
                            confidence -= 10
                            
                        party_name = "seller" if party_aspect["other"] == seller else "buyer"
                        return {
                            "found": True,
                            "favorable": True,
                            "confidence": confidence,
                            "reason": f"{translator_planet.value} translates light from {party_aspect['other'].value} ({party_name}) to {item.value} (item)",
                            "translator": translator_planet,
                            "pattern": "party_to_item"
                        }
        
        return {"found": False}
    
    def _check_enhanced_moon_testimony(self, chart: HoraryChart, querent: Planet, quesited: Planet,
                                     ignore_void_moon: bool = False) -> Dict[str, Any]:
        """Enhanced Moon testimony using the traditional void-of-course rule"""
        
        moon_pos = chart.planets[Planet.MOON]
        config = cfg()
        
        # ENHANCED: Check if Moon is void of course - now cautionary, not absolute blocker
        void_of_course = False
        void_reason = ""
        
        if not ignore_void_moon:
            void_check = self._is_moon_void_of_course_enhanced(chart)
            if void_check["void"] and not void_check["exception"]:
                void_of_course = True
                void_reason = void_check['reason']
        
        # ENHANCED: Continue with Moon analysis even if void (traditional cautionary approach)
        # Enhanced Moon analysis with accidental dignities
        phase_bonus = self._moon_phase_bonus(chart)
        speed_bonus = self._moon_speed_bonus(chart)
        angularity_bonus = self._moon_angularity_bonus(chart)
        
        total_moon_bonus = phase_bonus + speed_bonus + angularity_bonus
        adjusted_dignity = moon_pos.dignity_score + total_moon_bonus
        
        # ENHANCED: Check ALL Moon aspects to significators AND planets in target house (FIXED)
        moon_significator_aspects = []
        
        # Find quesited house number for planets-in-house testimony
        quesited_house_number = None
        for house_num, ruler in chart.house_rulers.items():
            if ruler == quesited:
                quesited_house_number = house_num
                break
        
        # Check all current Moon aspects
        for aspect in chart.aspects:
            if Planet.MOON in [aspect.planet1, aspect.planet2]:
                other_planet = aspect.planet2 if aspect.planet1 == Planet.MOON else aspect.planet1
                
                # Check if this is a significator aspect
                if other_planet in [querent, quesited]:
                    # Determine which house this planet rules
                    house_role = ""
                    if other_planet == querent:
                        house_role = "querent (L1)"
                    elif other_planet == quesited:
                        # Find which house this quesited planet rules
                        for house, ruler in chart.house_rulers.items():
                            if ruler == other_planet:
                                house_role = f"L{house}"
                                break
                        if not house_role:
                            house_role = "quesited"
                    
                    favorable = aspect.aspect in [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
                    aspect_desc = self._format_aspect_for_display("Moon", aspect.aspect.display_name, other_planet.value, aspect.applying)
                    
                    moon_significator_aspects.append({
                        "planet": other_planet,
                        "aspect": aspect.aspect,
                        "applying": aspect.applying,
                        "favorable": favorable,
                        "house_role": house_role,
                        "description": f"{aspect_desc} ({house_role})",
                        "testimony_type": "significator"
                    })
                
                # ADDED: Check Moon-to-benefic testimony (FIXED: missing benefic support detection)
                elif other_planet in [Planet.JUPITER, Planet.VENUS, Planet.SUN]:
                    favorable = aspect.aspect in [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
                    aspect_desc = self._format_aspect_for_display("Moon", aspect.aspect.display_name, other_planet.value, aspect.applying)
                    
                    moon_significator_aspects.append({
                        "planet": other_planet,
                        "aspect": aspect.aspect,
                        "applying": aspect.applying,
                        "favorable": favorable,
                        "house_role": f"benefic in {chart.planets[other_planet].house}th house",
                        "description": f"{aspect_desc} (Moon to benefic {other_planet.value})",
                        "testimony_type": "moon_to_benefic"
                    })
                
                # ADDED: Check planets-in-house testimony (Moon to planet located in quesited house)
                elif quesited_house_number and chart.planets[other_planet].house == quesited_house_number:
                    favorable = aspect.aspect in [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
                    aspect_desc = self._format_aspect_for_display("Moon", aspect.aspect.display_name, other_planet.value, aspect.applying)
                    
                    moon_significator_aspects.append({
                        "planet": other_planet,
                        "aspect": aspect.aspect,
                        "applying": aspect.applying,
                        "favorable": favorable,
                        "house_role": f"planet in {quesited_house_number}th house",
                        "description": f"{aspect_desc} (planet in {quesited_house_number}th house)",
                        "testimony_type": "planet_in_house"
                    })
        
        # If Moon has significant aspects to significators, prioritize this
        if moon_significator_aspects:
            applying_aspects = [a for a in moon_significator_aspects if a["applying"]]
            
            if applying_aspects:
                # FIXED: Sort by proximity to perfection (earliest first)
                # Find the degrees to exact for each aspect
                applying_with_degrees = []
                for aspect_data in applying_aspects:
                    # Find corresponding aspect in chart.aspects to get degrees_to_exact
                    for chart_aspect in chart.aspects:
                        if (Planet.MOON in [chart_aspect.planet1, chart_aspect.planet2] and
                            aspect_data["planet"] in [chart_aspect.planet1, chart_aspect.planet2] and
                            chart_aspect.aspect == aspect_data["aspect"] and
                            chart_aspect.applying == aspect_data["applying"]):
                            
                            applying_with_degrees.append({
                                **aspect_data,
                                "degrees_to_exact": chart_aspect.degrees_to_exact
                            })
                            break
                
                # Sort by degrees to exact (earliest perfection first)
                applying_with_degrees.sort(key=lambda x: x.get("degrees_to_exact", 999))
                primary_aspect = applying_with_degrees[0]  # Earliest perfection
                favorable = primary_aspect["favorable"]
                
                all_descriptions = [a["description"] for a in applying_aspects]
                reason = f"Moon testimony: {', '.join(all_descriptions)}"
                
                # Calculate confidence based on moon condition and aspects
                base_confidence = config.confidence.lunar_confidence_caps.favorable if favorable else config.confidence.lunar_confidence_caps.unfavorable
                if void_of_course:
                    base_confidence = min(base_confidence, config.confidence.lunar_confidence_caps.neutral)
                
                return {
                    "favorable": favorable,
                    "unfavorable": not favorable,
                    "reason": reason,
                    "supportive": True,  # Marks this as significant Moon testimony
                    "timing": "Within days" if applying_aspects else "Variable",
                    "void_of_course": void_of_course,
                    "aspects": moon_significator_aspects,
                    "confidence": base_confidence
                }
        
        # FIXED: Moon's next aspect should be PRIMARY, not fallback (traditional horary priority)
        next_aspect = chart.moon_next_aspect
        if next_aspect and next_aspect.planet in [querent, quesited]:
            other_planet = next_aspect.planet
            aspect_type = next_aspect.aspect
            favorable = aspect_type in [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
            
            # Calculate confidence for next aspect case
            base_confidence = config.confidence.lunar_confidence_caps.favorable if favorable else config.confidence.lunar_confidence_caps.unfavorable
            if void_of_course:
                base_confidence = min(base_confidence, config.confidence.lunar_confidence_caps.neutral)
            
            return {
                "favorable": favorable,
                "unfavorable": not favorable,
                "reason": f"Moon next {aspect_type.display_name}s {other_planet.value} (total dignity: {adjusted_dignity:+d})",
                "timing": next_aspect.perfection_eta_description,
                "void_of_course": False,
                "confidence": base_confidence
            }
        
        # ENHANCED: Moon's general condition with void status as cautionary modifier
        base_reason = ""
        favorable = False
        unfavorable = False
        
        if adjusted_dignity > 0:
            favorable = True
            base_reason = f"Moon well-dignified in {moon_pos.sign.sign_name} (adjusted dignity: {adjusted_dignity:+d})"
        elif adjusted_dignity < -3:
            unfavorable = True  
            base_reason = f"Moon poorly dignified in {moon_pos.sign.sign_name} (adjusted dignity: {adjusted_dignity:+d})"
        else:
            base_reason = f"Moon testimony neutral (adjusted dignity: {adjusted_dignity:+d})"
        
        # Add void of course as cautionary note, not blocking factor
        if void_of_course:
            if favorable:
                # Void reduces favorable Moon but doesn't negate it
                base_reason += f" - BUT Moon void of course ({void_reason}) - reduces effectiveness"
            else:
                base_reason += f" - Moon void of course ({void_reason})"
        
        # Calculate confidence for general moon testimony
        if favorable:
            base_confidence = config.confidence.lunar_confidence_caps.favorable
        elif unfavorable:
            base_confidence = config.confidence.lunar_confidence_caps.unfavorable
        else:
            base_confidence = config.confidence.lunar_confidence_caps.neutral
            
        if void_of_course:
            base_confidence = min(base_confidence, config.confidence.lunar_confidence_caps.neutral)
        
        return {
            "favorable": favorable,
            "unfavorable": unfavorable,
            "reason": base_reason,
            "void_of_course": void_of_course,
            "void_caution": void_of_course,  # Flag for main judgment logic
            "confidence": base_confidence
        }
    
    def _format_aspect_for_display(self, planet1: str, aspect_data, planet2: str, applying: bool) -> str:
        """Format aspect for display in frontend-compatible style"""
        
        # Extract aspect name from tuple (0, "conjunction", "Conjunction")
        if isinstance(aspect_data, tuple) and len(aspect_data) >= 3:
            aspect_name = aspect_data[2]  # Get "Conjunction" from tuple
        else:
            aspect_name = str(aspect_data)  # Fallback for strings
        
        # Convert aspect names to symbols (matching frontend)
        aspect_symbols = {
            'Conjunction': '☌',
            'Sextile': '⚹', 
            'Square': '□',
            'Trine': '△',
            'Opposition': '☍'
        }
        
        # Get aspect symbol or fallback
        symbol = aspect_symbols.get(aspect_name, '○')
        
        # Format status
        status = "applying" if applying else "separating"
        
        # Return formatted string matching frontend style: "Planet1 ☌ Planet2 (applying)"
        return f"{planet1} {symbol} {planet2} ({status})"
    
    def _get_aspect_symbol(self, aspect_data) -> str:
        """Get aspect symbol from aspect data"""
        # Extract aspect name from tuple (0, "conjunction", "Conjunction")
        if isinstance(aspect_data, tuple) and len(aspect_data) >= 3:
            aspect_name = aspect_data[2]  # Get "Conjunction" from tuple
        else:
            aspect_name = str(aspect_data)  # Fallback for strings
        
        # Convert aspect names to symbols
        aspect_symbols = {
            'Conjunction': '☌',
            'Sextile': '⚹', 
            'Square': '□',
            'Trine': '△',
            'Opposition': '☍'
        }
        
        return aspect_symbols.get(aspect_name, '○')
    
    def _check_benefic_aspects_to_significators(self, chart: HoraryChart, querent_planet: Planet, quesited_planet: Planet) -> Dict[str, Any]:
        """ENHANCED: Check for beneficial aspects to significators (traditional hierarchy)"""

        # Traditional benefics excluding the Sun (handled via R28 luminary rule)
        benefics = [Planet.JUPITER, Planet.VENUS]
        significators = [querent_planet, quesited_planet]

        # Determine which house the quesited planet rules
        quesited_house_number = None
        for house_num, ruler in chart.house_rulers.items():
            if ruler == quesited_planet:
                quesited_house_number = house_num
                break

        benefic_aspects = []
        total_score = 0
        separating_notes: List[str] = []

        for benefic in benefics:
            if benefic in significators or benefic not in chart.planets:
                continue  # Skip if benefic IS a significator or missing

            benefic_pos = chart.planets[benefic]

            for significator in significators:
                if significator not in chart.planets:
                    continue

                # Find aspects between benefic and significator
                for aspect in chart.aspects:
                    if ((aspect.planet1 == benefic and aspect.planet2 == significator) or
                        (aspect.planet1 == significator and aspect.planet2 == benefic)):

                        if not aspect.applying:
                            # Record separating aspects as historical notes only
                            separating_notes.append(
                                self._format_aspect_for_display(
                                    benefic.value,
                                    aspect.aspect.display_name,
                                    significator.value,
                                    aspect.applying,
                                )
                            )
                            continue

                        # Calculate benefic strength
                        aspect_strength = self._calculate_benefic_aspect_strength(
                            benefic, significator, aspect, chart)

                        if aspect_strength > 0:
                            description = self._format_aspect_for_display(
                                benefic.value, aspect.aspect.display_name,
                                significator.value, aspect.applying)
                            benefic_aspects.append({
                                "benefic": benefic.value,
                                "significator": significator.value,
                                "aspect": aspect.aspect.display_name,
                                "applying": aspect.applying,
                                "degrees": aspect.degrees_to_exact,
                                "strength": aspect_strength,
                                "house_position": benefic_pos.house,
                                "description": description,
                                "type": "aspect",
                            })
                            total_score += aspect_strength

        # Check for benefic planets located in the quesited's house
        if quesited_house_number:
            for benefic in benefics:
                if benefic in significators or benefic not in chart.planets:
                    continue

                benefic_pos = chart.planets[benefic]
                if benefic_pos.house == quesited_house_number:
                    strength = 0
                    if benefic_pos.house in [1, 4, 7, 10]:
                        strength += 4
                    elif benefic_pos.house in [2, 5, 8, 11]:
                        strength += 2
                    else:
                        strength += 1

                    if benefic_pos.dignity_score > 0:
                        strength += min(3, benefic_pos.dignity_score)

                    if strength > 0:
                        desc = f"{benefic.value} in {quesited_house_number}th house"
                        if benefic_pos.house in [1, 4, 7, 10]:
                            desc += " (angular)"
                        if benefic_pos.dignity_score > 0:
                            desc += f" ({benefic_pos.dignity_score:+d} dignity)"

                        benefic_aspects.append({
                            "benefic": benefic.value,
                            "significator": None,
                            "aspect": None,
                            "applying": None,
                            "degrees": None,
                            "strength": strength,
                            "house_position": benefic_pos.house,
                            "description": desc,
                            "type": "house",
                        })
                        total_score += strength

        if benefic_aspects:
            # Determine result based on total score
            if total_score >= 15:  # Strong benefic support
                result = "YES"
                confidence = min(85, 60 + total_score)
            elif total_score >= 8:   # Moderate benefic support
                result = "YES"
                confidence = min(75, 55 + total_score)
            else:                    # Weak but positive
                result = "UNCLEAR"
                confidence = 50 + total_score

            strongest = max(benefic_aspects, key=lambda x: x["strength"])

            reason = strongest.get("description", "")
            if separating_notes:
                reason += f"; separating aspects noted: {', '.join(separating_notes)}"
            return {
                "favorable": result == "YES",
                "neutral": result == "UNCLEAR",
                "unfavorable": False,
                "confidence": confidence,
                "total_score": total_score,
                "aspects": benefic_aspects,
                "strongest_aspect": strongest,
                "reason": reason,
            }
        else:
            reason = "No benefic support to significators"
            if separating_notes:
                reason += f" (separating aspects noted: {', '.join(separating_notes)})"
            return {
                "favorable": False,
                "neutral": False,
                "unfavorable": False,
                "confidence": None,  # FIX: Don't override hybrid confidence calculation with 0
                "total_score": 0,
                "aspects": [],
                "reason": reason,
            }

    def _calculate_benefic_aspect_strength(self, benefic: Planet, significator: Planet, aspect: AspectInfo, chart: HoraryChart, relevant_planets: set = None) -> int:
        """Calculate strength of benefic aspect to significator (topic-agnostic)"""
        
        # Check relevance: only count if aspect touches L1/quesited/relevant set
        if relevant_planets:
            p1, p2 = aspect.planet1, aspect.planet2
            if not ({p1, p2} & relevant_planets):
                return 0  # Not relevant to decision
        
        base_strength = 0
        benefic_pos = chart.planets[benefic]
        
        # Enhanced aspect type scoring (neutral/negative for hard aspects unless rescued)
        if aspect.aspect == Aspect.TRINE:
            base_strength = 12
        elif aspect.aspect == Aspect.SEXTILE:
            base_strength = 8
        elif aspect.aspect == Aspect.CONJUNCTION:
            base_strength = 6   # Generally positive but context-dependent
        elif aspect.aspect == Aspect.SQUARE:
            base_strength = -2  # Negative by default (rescue via reception)
        elif aspect.aspect == Aspect.OPPOSITION:
            base_strength = -4  # More challenging
        else:
            base_strength = 0
            
        # Applying vs separating
        if not aspect.applying:
            return 0  # Separating aspects provide no benefic support
        if base_strength > 0:
            base_strength += 2  # Bonus for positive aspects
        else:
            base_strength -= 1  # Penalty for negative aspects
            
        # Reception rescue for hard aspects
        if base_strength < 0:
            reception_strength = self._get_reception_strength(benefic, significator, chart)
            if reception_strength >= 3:  # Strong reception rescues hard aspects
                base_strength += 3  # Partial rescue, not full flip
                
        # Retrograde dampening
        if benefic_pos.retrograde:
            if base_strength > 0:
                base_strength = max(1, base_strength - 3)  # Reduce positive strength
            else:
                base_strength -= 2  # Worsen negative strength
            
        # Closeness bonus (only for positive aspects)
        if base_strength > 0:
            if aspect.degrees_to_exact <= 3:
                base_strength += 2
            elif aspect.degrees_to_exact <= 6:
                base_strength += 1
                
        # House position bonus (only for positive aspects)
        if base_strength > 0:
            if benefic_pos.house in [1, 4, 7, 10]:
                base_strength += 2  # Reduced angular bonus
            elif benefic_pos.house in [2, 5, 8, 11]:
                base_strength += 1  # Reduced succeedent bonus
                
        # Benefic planet bonuses (only for positive aspects)
        if base_strength > 0:
            if benefic == Planet.JUPITER:
                base_strength += 1  # Reduced greater benefic bonus
            elif benefic == Planet.VENUS:
                base_strength += 1  # Lesser benefic bonus
                    
        # Dignity bonus (only for positive aspects)
        if base_strength > 0 and benefic_pos.dignity_score > 0:
            base_strength += min(2, benefic_pos.dignity_score // 2)
            
        return max(-6, min(base_strength, 15))  # Allow negative, cap positive
    
    def _get_reception_strength(self, planet1: Planet, planet2: Planet, chart: HoraryChart) -> int:
        """Get numerical reception strength between two planets"""
        reception_calc = self.reception_calculator.calculate_comprehensive_reception(chart, planet1, planet2)
        return reception_calc.get("traditional_strength", 0)
    
    def _is_moon_void_of_course_enhanced(self, chart: HoraryChart) -> Dict[str, Any]:
        """Traditional void of course check - GROUND TRUTH implementation
        
        The Moon is void of course if, before leaving its current sign, it will not apply 
        to any of the seven classical planets (Sun, Mercury, Venus, Mars, Jupiter, Saturn) 
        by a Ptolemaic aspect (conjunction, sextile, square, trine, opposition) within the 
        permitted orb, considering true motion (including retrograde).
        """
        
        return self._void_traditional_ground_truth(chart)
    
    def _void_traditional_ground_truth(self, chart: HoraryChart) -> Dict[str, Any]:
        """GROUND TRUTH: Traditional void of course implementation
        
        Moon is void if it will not apply to any classical planet by Ptolemaic aspect
        within permitted orb before leaving its current sign.
        """
        moon_pos = chart.planets[Planet.MOON]
        config = cfg()

        # Calculate degrees left in current sign
        moon_degree_in_sign = moon_pos.longitude % 30
        degrees_left_in_sign = 30 - moon_degree_in_sign

        # Check if Moon is stationary (cannot be void if not moving)
        if abs(moon_pos.speed) < config.timing.stationary_speed_threshold:
            return {
                "void": False,
                "exception": False,
                "reason": "Moon stationary - cannot be void of course",
                "degrees_left_in_sign": degrees_left_in_sign,
                "first_applying_aspect": None,
            }

        # Determine next lunar aspect using analytic helper
        moon_next_aspect = calculate_moon_next_aspect(
            chart.planets,
            chart.julian_day,
            ignore_orb_for_voc=True,
        )

        if moon_next_aspect is None:
            # CRITICAL FIX: Lilly's sign-based VOC nuance
            # VOC is less severe in Taurus, Cancer, Sagittarius, Pisces
            mitigating_signs = {"Taurus", "Cancer", "Sagittarius", "Pisces"}
            sign_mitigation = moon_pos.sign.sign_name in mitigating_signs
            
            return {
                "void": True,
                "exception": sign_mitigation,  # Exception flag for mitigating signs
                "reason": f"Moon makes no applying aspects before leaving {moon_pos.sign.sign_name}" + 
                         (" (mitigated in Lilly-favorable sign)" if sign_mitigation else ""),
                "degrees_left_in_sign": degrees_left_in_sign,
                "first_applying_aspect": None,
                "sign_mitigation": sign_mitigation,
            }

        return {
            "void": False,
            "exception": False,
            "reason": (
                f"Moon will {moon_next_aspect.aspect.display_name.lower()} "
                f"{moon_next_aspect.planet.value} in {moon_next_aspect.perfection_eta_days:.1f} days"
            ),
            "degrees_left_in_sign": degrees_left_in_sign,
            "first_applying_aspect": moon_next_aspect,
        }
    
    # REMOVED: Old configurable void methods replaced with ground truth implementation
    
    # REMOVED: All configurable void methods - using only ground truth implementation
    
    def _calculate_aspect_positions(self, planet_longitude: float, aspect: Aspect, moon_sign: Sign) -> List[float]:
        """Calculate aspect positions (preserved from original)"""
        positions = []
        
        aspect_positions = [
            (planet_longitude + aspect.degrees) % 360,
            (planet_longitude - aspect.degrees) % 360
        ]
        
        sign_start = moon_sign.start_degree
        sign_end = (sign_start + 30) % 360
        
        for pos in aspect_positions:
            pos_normalized = pos % 360
            
            if sign_start < sign_end:
                if sign_start <= pos_normalized < sign_end:
                    positions.append(pos_normalized)
            else:  
                if pos_normalized >= sign_start or pos_normalized < sign_end:
                    positions.append(pos_normalized)
        
        return positions
    
    def _build_moon_story(self, chart: HoraryChart) -> List[Dict]:
        """Enhanced Moon story with real timing calculations"""
        
        moon_pos = chart.planets[Planet.MOON]
        moon_speed = self.calculator.get_real_moon_speed(chart.julian_day)
        
        # Calculate how much time Moon has left in current sign
        moon_degree_in_sign = moon_pos.longitude % 30
        degrees_left_in_sign = 30 - moon_degree_in_sign
        days_to_sign_exit = degrees_left_in_sign / abs(moon_speed) if abs(moon_speed) > 0 else None
        
        # Get current aspects
        current_moon_aspects = []
        for aspect in chart.aspects:
            if Planet.MOON in [aspect.planet1, aspect.planet2]:
                other_planet = aspect.planet2 if aspect.planet1 == Planet.MOON else aspect.planet1
                
                # Enhanced timing using analytic solver
                if aspect.applying:
                    other_pos = chart.planets[other_planet]
                    timing_days = self._calculate_future_aspect_time(
                        moon_pos,
                        other_pos,
                        aspect.aspect,
                        chart.julian_day,
                        cfg().timing.max_future_days,
                    ) or 0
                    
                    # Check if aspect will perfect before Moon changes signs (CONSISTENCY FIX)
                    will_perfect_in_sign = True
                    if days_to_sign_exit is not None and timing_days > days_to_sign_exit:
                        will_perfect_in_sign = False
                    
                    # Only include applying aspects that will perfect within the current sign
                    if will_perfect_in_sign:
                        timing_estimate = self._format_timing_description_enhanced(timing_days)
                        current_moon_aspects.append({
                            "planet": other_planet.value,
                            "aspect": aspect.aspect.display_name,
                            "orb": float(aspect.orb),
                            "applying": bool(aspect.applying),
                            "status": "applying" if aspect.applying else "separating",
                            "timing": str(timing_estimate),
                            "days_to_perfect": float(timing_days)
                        })
                else:
                    # Always include separating aspects (they've already happened)
                    current_moon_aspects.append({
                        "planet": other_planet.value,
                        "aspect": aspect.aspect.display_name,
                        "orb": float(aspect.orb),
                        "applying": bool(aspect.applying),
                        "status": "applying" if aspect.applying else "separating",
                        "timing": "Past",
                        "days_to_perfect": 0.0
                    })
        
        # Sort by timing for applying aspects, orb for separating
        current_moon_aspects.sort(key=lambda x: x.get("days_to_perfect", 999) if x["applying"] else x["orb"])
        
        return current_moon_aspects
    
    def _format_timing_description_enhanced(self, days: float) -> str:
        """Enhanced timing description with configuration"""
        if days < 0.5:
            return "Within hours"
        elif days < 1:
            return "Within a day"
        elif days < 7:
            return f"Within {int(days)} days"
        elif days < 30:
            return f"Within {int(days/7)} weeks"
        elif days < 365:
            return f"Within {int(days/30)} months"
        else:
            return "More than a year"
    
    def _calculate_enhanced_timing(self, chart: HoraryChart, perfection: Dict) -> str:
        """Enhanced timing calculation with real Moon speed"""
        
        if "aspect" in perfection:
            aspect_data = perfection["aspect"]
            
            # Handle both dictionary format and Aspect object format
            if isinstance(aspect_data, dict):
                degrees = aspect_data["degrees_to_exact"]
            else:
                # Handle Aspect object - get degrees_to_exact from perfection root level
                degrees = perfection.get("degrees_to_exact") or perfection.get("t_perfect_days", 0) * 13.0  # Fallback calculation
                
            moon_speed = self.calculator.get_real_moon_speed(chart.julian_day)
            if degrees > 0 and moon_speed > 0:
                timing_days = degrees / moon_speed
                return self._format_timing_description_enhanced(timing_days)
        
        # Check for other timing indicators
        if "t_perfect_days" in perfection:
            return self._format_timing_description_enhanced(perfection["t_perfect_days"])
            
        return "Timing uncertain"
    
    # Preserve all existing helper methods for backward compatibility
    def _identify_significators(self, chart: HoraryChart, question_analysis: Dict) -> Dict[str, Any]:
        """Identify significators using shared taxonomy resolver."""

        category = question_analysis.get("question_type")
        manual_houses = question_analysis.get("relevant_houses")
        significator_info = question_analysis.get("significators", {})
        return resolve_significators(
            chart,
            category,
            manual_houses=manual_houses,
            significator_info=significator_info,
        )

    def _find_applying_aspect(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> Optional[Dict]:
        """Find applying aspect between two planets (preserved)"""
        for aspect in chart.aspects:
            if ((aspect.planet1 == planet1 and aspect.planet2 == planet2) or
                (aspect.planet1 == planet2 and aspect.planet2 == planet1)) and aspect.applying:
                return {
                    "aspect": aspect.aspect,
                    "orb": aspect.orb,
                    "degrees_to_exact": aspect.degrees_to_exact,
                    "applying": True
                }
        return None
    
    def _find_any_aspect(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> Optional[Dict]:
        """Find any aspect (applying or separating) between two planets"""
        for aspect in chart.aspects:
            if ((aspect.planet1 == planet1 and aspect.planet2 == planet2) or
                (aspect.planet1 == planet2 and aspect.planet2 == planet1)):
                return {
                    "aspect": aspect.aspect,
                    "orb": aspect.orb,
                    "degrees_to_exact": aspect.degrees_to_exact,
                    "applying": aspect.applying,
                    "full_aspect": aspect  # Include full aspect for detailed analysis
                }
        return None
    
    def _check_enhanced_perfection(self, chart: HoraryChart, querent: Planet, quesited: Planet,
                                 exaltation_confidence_boost: float = 15.0, window_days: int = None) -> Dict[str, Any]:
        """Enhanced perfection check with configuration"""
        
        config = cfg()
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]
        
        # 1. Enhanced direct aspect analysis - Check for ANY aspect first
        any_direct_aspect = self._find_any_aspect(chart, querent, quesited)
        applying_direct_aspect = self._find_applying_aspect(chart, querent, quesited)
        direct_aspect_found = False
        direct_aspect_applying = False
        
        # If there's an applying direct aspect, process it normally
        if applying_direct_aspect:
            direct_aspect = applying_direct_aspect
            direct_aspect_found = True
            direct_aspect_applying = True
            
            # CRITICAL FIX: Check if this is a combustion conjunction (denial, not perfection)
            is_combustion_conjunction = False
            if direct_aspect["aspect"] == Aspect.CONJUNCTION:
                # Check if one planet is the Sun and the other is combust
                sun_planet = None
                other_planet = None
                
                if querent == Planet.SUN:
                    sun_planet = querent
                    other_planet = quesited
                elif quesited == Planet.SUN:
                    sun_planet = quesited
                    other_planet = querent
                
                if sun_planet and other_planet:
                    other_pos = chart.planets[other_planet]
                    if hasattr(other_pos, 'solar_condition') and other_pos.solar_condition.condition == "Combustion":
                        is_combustion_conjunction = True
                        return {
                            "perfects": False,
                            "type": "combustion_denial",
                            "favorable": False,
                            "confidence": 85,
                            "reason": f"Combustion denial: {other_planet.value} conjunct Sun causes combustion, not perfection",
                            "reception": self._detect_reception_between_planets(chart, querent, quesited),
                            "aspect": direct_aspect
                        }
            
            perfects_in_sign, impediment = self._enhanced_perfects_in_sign(querent_pos, quesited_pos, direct_aspect, chart)

            if impediment and impediment.get("type") == "refranation":
                planet = impediment["planet"].value
                return {
                    "perfects": False,
                    "type": "refranation",
                    "favorable": False,
                    "confidence": cfg().confidence.denial.refranation,
                    "reason": f"{planet} stations before perfecting aspect",
                    "aspect": direct_aspect,
                }

            if perfects_in_sign and not is_combustion_conjunction:
                reception_info = self.reception_calculator.calculate_comprehensive_reception(
                    chart, querent, quesited
                )
                reception = reception_info["type"]
                mutual = reception_info.get("mutual", "none")
                one_way = reception_info.get("one_way", [])

                # Enhanced reception weighting with configuration
                if mutual == "mutual_rulership":
                    return {
                        "perfects": True,
                        "type": "direct",
                        "favorable": True,
                        "confidence": config.confidence.perfection.direct_with_mutual_rulership,
                        "reason": f"Direct perfection: {self._format_aspect_for_display(querent.value, direct_aspect['aspect'], quesited.value, True)} with {reception_info['display_text']}",
                        "reception": reception,
                        "aspect": direct_aspect,
                        "tags": [{"family": "perfection", "kind": "direct"}],
                    }
                elif mutual == "mutual_exaltation":
                    base_confidence = config.confidence.perfection.direct_with_mutual_exaltation
                    boosted_confidence = min(100, base_confidence + exaltation_confidence_boost)

                    return {
                        "perfects": True,
                        "type": "direct",
                        "favorable": True,
                        "confidence": int(boosted_confidence),
                        "reason": f"Direct perfection: {self._format_aspect_for_display(querent.value, direct_aspect['aspect'], quesited.value, True)} with {reception_info['display_text']}",
                        "reception": reception,
                        "aspect": direct_aspect,
                        "tags": [{"family": "perfection", "kind": "direct"}],
                    }
                else:
                    favorable, penalty_reasons, one_way_flag = self._is_aspect_favorable_enhanced(
                        direct_aspect["aspect"], reception_info, chart, querent, quesited
                    )

                    aspect_name = direct_aspect['aspect'].display_name
                    base_reason = f"{aspect_name} between significators"

                    if favorable:
                        confidence_value = config.confidence.perfection.direct_basic
                        if penalty_reasons:
                            penalty = 15
                            if one_way_flag:
                                penalty = 5
                                penalty_reasons.append("one-way reception softens difficulties")
                            confidence_value = max(confidence_value - penalty, 0)
                            base_reason = f"{aspect_name} between significators but weakened: {', '.join(penalty_reasons)}"
                        if mutual != "none":
                            reception_bonus = getattr(
                                config.confidence.reception, f"{mutual}_bonus", 5
                            )
                            confidence_value = min(confidence_value + reception_bonus, 100)
                            base_reason = f"{base_reason} with {reception_info['display_text']}"
                        elif one_way:
                            reception_bonus = getattr(
                                config.confidence.reception, "one_way_bonus", 3
                            )
                            confidence_value = min(confidence_value + reception_bonus, 100)
                            base_reason = f"{base_reason} with {reception_info['display_text']}"
                        return {
                            "perfects": True,
                            "type": "direct",
                            "favorable": True,
                            "confidence": confidence_value,
                            "reason": base_reason,
                            "reception": reception,
                            "aspect": direct_aspect,
                            "tags": [{"family": "perfection", "kind": "direct"}],
                        }
                    else:
                        if penalty_reasons:
                            base_reason = f"{aspect_name} penalized: {'; '.join(penalty_reasons)}"
                        elif reception == "none":
                            base_reason = f"{aspect_name} lacks reception"
                        else:
                            base_reason = f"{aspect_name} unfavorable"

                        return {
                            "perfects": True,
                            "type": "direct_penalized",
                            "favorable": False,
                            "confidence": max(config.confidence.perfection.direct_basic - 25, 0),
                            "reason": base_reason,
                            "reception": reception,
                            "aspect": direct_aspect,
                            "tags": [{"family": "perfection", "kind": "direct"}],
                        }
        
        # 2. House placement perfection (planets in quesited house aspecting house ruler)
        house_perfection = self._check_house_placement_perfection(chart, querent, quesited, window_days)
        if house_perfection and house_perfection.get("perfects", False):
            return house_perfection
        
        # 3. Direct timed perfection (future perfection within window) - ALWAYS CHECK when timeframe provided
        if window_days is not None:
            direct_timed = self._check_direct_timed_perfection(chart, querent, quesited, window_days)
            if direct_timed and direct_timed.get("perfects", False):
                return direct_timed
        
        # 3. Primary Translation of Light check (independent perfection route)
        # CRITICAL FIX: Check both translation methods and prioritize by timing
        translation_result = None
        translation = self._check_enhanced_translation_of_light(chart, querent, quesited)
        
        # CRITICAL FIX: Also check for Moon translation using chart data
        if not (translation and translation.get("found", False)):
            translation = self._check_moon_translation_pattern(chart, querent, quesited)
        
        if translation and translation.get("found", False):
            # Build enhanced reason text with traditional interpretation
            base_reason = f"Translation of light by {translation['translator'].value} - {translation['sequence']}"
            challenge_reasons = translation.get("challenge_reasons", [])
            
            if challenge_reasons:
                challenge_text = ", ".join(challenge_reasons)
                enhanced_reason = f"{base_reason} - Success through challenges ({challenge_text})"
            else:
                enhanced_reason = f"{base_reason} - Clear connection between significators"
            
            translation_result = {
                "perfects": True,
                "type": "translation",
                "favorable": translation["favorable"],
                "confidence": max(translation.get("confidence", 50), 40),
                "reason": enhanced_reason,
                "translator": translation["translator"],
                "challenge_reasons": challenge_reasons,
                "negative_reasons": translation.get("negative_reasons", []),
                "tags": [{"family": "perfection", "kind": "tol"}],
                "timing_days": translation.get("timing_days", 0.0)
            }

        # 4. Primary Collection of Light check (independent perfection route)
        collection_result = None
        collection = self._check_enhanced_collection_of_light(chart, querent, quesited)
        
        # CRITICAL FIX: Also check for simple collection pattern without strict requirements
        if not (collection and collection.get("found", False)):
            collection = self._check_simple_collection_pattern(chart, querent, quesited)
        
        if collection and collection.get("found", False):
            # Build enhanced reason text with traditional interpretation
            base_reason = f"Collection of light by {collection['collector'].value}"
            challenge_reasons = collection.get("challenge_reasons", [])
            
            if challenge_reasons:
                challenge_text = ", ".join(challenge_reasons)
                enhanced_reason = f"{base_reason} - Success through challenges ({challenge_text})"
            else:
                enhanced_reason = base_reason
            
            collection_result = {
                "perfects": True,
                "type": "collection",
                "favorable": collection["favorable"],
                "confidence": max(collection.get("confidence", 50), 40),  # Ensure minimum confidence
                "reason": enhanced_reason,
                "collector": collection["collector"],
                "challenge_reasons": challenge_reasons,  # New field for challenges
                "negative_reasons": collection.get("negative_reasons", []),  # Backward compatibility
                "candidates": collection.get("candidates"),
                "tags": [{"family": "perfection", "kind": "col"}],
                "timing_days": collection.get("timing_days", float('inf'))
            }
        
        # 5. CRITICAL FIX: Primary perfection resolver - compare timing if both exist
        if translation_result and collection_result:
            translation_timing = translation_result.get("timing_days", float('inf'))
            collection_timing = collection_result.get("timing_days", float('inf'))
            
            if translation_timing <= collection_timing:
                # Translation is earlier or same time - make it primary
                primary_result = translation_result.copy()
                primary_result["secondary_perfection"] = {
                    "type": "collection", 
                    "reason": collection_result["reason"],
                    "timing_days": collection_timing
                }
                return primary_result
            else:
                # Collection is earlier - make it primary
                primary_result = collection_result.copy()
                primary_result["secondary_perfection"] = {
                    "type": "translation",
                    "reason": translation_result["reason"], 
                    "timing_days": translation_timing
                }
                return primary_result
        elif translation_result:
            return translation_result
        elif collection_result:
            return collection_result
        
        # 6. Handle separating direct aspects (fallback when no perfection routes work)
        if any_direct_aspect and not applying_direct_aspect:
            aspect_name = self._format_aspect_for_display(
                querent.value, any_direct_aspect['aspect'], quesited.value, False
            )
            
            return {
                "perfects": False,
                "type": "separating_direct",
                "favorable": False,
                "confidence": 75,  # High confidence in NO for separating aspects
                "reason": f"Separating {aspect_name} - opportunity has passed",
                "reception": self._detect_reception_between_planets(chart, querent, quesited),
                "aspect": any_direct_aspect,
                "tags": [{"family": "perfection", "kind": "separating_direct"}],
            }
        
        # 4. Enhanced mutual reception without aspect (GATED - no standalone bonuses)
        reception = self._check_enhanced_mutual_reception(chart, querent, quesited)
        if reception in ["mutual_rulership", "mutual_exaltation"]:
            # Reception noted but not applied as standalone bonus
            return {
                "perfects": False,
                "type": "none",  # CRITICAL FIX: Don't use special type that bypasses hybrid confidence
                "favorable": False,  # Cannot be favorable without perfection
                "confidence": 0,     # No confidence bonus without perfection 
                "reason": f"Reception: {self._format_reception_for_display(reception, querent, quesited, chart)} - noted but requires perfection",
                "reception": reception,
                "reception_support_present_but_no_perfection": True
            }
        
        return {
            "perfects": False,
            "reason": "No perfection found between significators"
        }
    
    def _check_direct_timed_perfection(self, chart: HoraryChart, querent: Planet, quesited: Planet, window_days: int) -> Dict[str, Any]:
        """Check for future direct perfection between significators within timeframe window"""
        
        config = cfg()
        max_window = min(window_days, getattr(config.timing, "max_window_days", 365))

        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]

        # Ensure reception context is always available for confidence adjustments
        reception_info = {
            "type": "none",
            "mutual": "none",
            "one_way": [],
        }
        try:
            # Use centralized comprehensive reception calculator if available
            reception_info = self.reception_calculator.calculate_comprehensive_reception(
                chart, querent, quesited
            ) or reception_info
        except Exception:
            # Fallback to legacy detector returning a simple string
            try:
                r_type = self._detect_reception_between_planets(chart, querent, quesited)
                reception_info["type"] = r_type or "none"
            except Exception:
                pass
        # Safe defaults
        mutual = reception_info.get("mutual", "none")
        one_way = reception_info.get("one_way", [])

        # Moiety-based orb limit
        moiety_q = getattr(config.orbs.moieties, querent.value, self._get_planet_moiety(querent))
        moiety_qs = getattr(config.orbs.moieties, quesited.value, self._get_planet_moiety(quesited))
        orb_limit = moiety_q + moiety_qs

        target_angles = {
            Aspect.CONJUNCTION: 0,
            Aspect.SEXTILE: 60,
            Aspect.SQUARE: 90,
            Aspect.TRINE: 120,
            Aspect.OPPOSITION: 180,
        }

        alignment_info = None
        
        # First check if there's an existing aspect between significators
        existing_aspect = None
        for aspect_info in chart.aspects:
            if ((aspect_info.planet1 == querent and aspect_info.planet2 == quesited) or
                (aspect_info.planet1 == quesited and aspect_info.planet2 == querent)):
                existing_aspect = aspect_info
                break
        
        # Calculate future aspect perfection times
        aspect_types = [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.SQUARE, Aspect.TRINE, Aspect.OPPOSITION]
        
        for aspect_type in aspect_types:
            # If there's an existing aspect of this type that's separating, skip it
            if (
                existing_aspect
                and existing_aspect.aspect == aspect_type
                and not existing_aspect.applying
            ):
                continue  # Can't have future perfection of same aspect type that's already separating

            target_angle = target_angles[aspect_type]
            delta = (quesited_pos.longitude + target_angle - querent_pos.longitude) % 360.0
            if delta > 180:
                delta -= 360.0
            separation = abs(delta)
            derivative = quesited_pos.speed - querent_pos.speed
            applying = derivative * math.copysign(1, delta) < 0
            if separation > orb_limit or not applying:
                continue

            days_to_perfection = self._calculate_future_aspect_time(
                querent_pos, quesited_pos, aspect_type, chart.julian_day, max_window
            )

            if days_to_perfection is not None and 0 < days_to_perfection <= max_window:
                # Sign boundary check
                if getattr(config.perfection, "require_in_sign", False):
                    exit_q = days_to_sign_exit(querent_pos.longitude, querent_pos.speed)
                    exit_qs = days_to_sign_exit(quesited_pos.longitude, quesited_pos.speed)
                    exits = [e for e in (exit_q, exit_qs) if e is not None]
                    if exits and days_to_perfection > min(exits):
                        return {
                            "perfects": False,
                            "type": "out_of_sign",
                            "reason": "Perfection occurs after sign exit",
                        }

                max_horary_days = getattr(config.timing, "max_horary_days", 30)
                if days_to_perfection > max_horary_days:
                    alignment_info = {
                        "perfects": False,
                        "type": "astronomical_alignment",
                        "reason": f"Future {aspect_type.display_name} between {querent.value} and {quesited.value} in {days_to_perfection:.1f} days (astronomical alignment)",
                        "t_perfect_days": days_to_perfection,
                        "aspect": aspect_type,
                        "tags": [{"family": "perfection", "kind": "astronomical_alignment"}],
                    }
                    continue

                # Check for prohibitions/refranations before perfection
                prohibition_check = self._check_future_prohibitions(
                    chart, querent, quesited, days_to_perfection
                )
                if prohibition_check.get("prohibited"):
                    # Pass through prohibitor details so callers know what blocked perfection
                    return {
                        "perfects": False,
                        "type": prohibition_check.get("type", "prohibition"),
                        "reason": prohibition_check["reason"],
                        "prohibitor": prohibition_check.get("prohibitor"),
                        "significator": prohibition_check.get("significator"),
                    }

                if prohibition_check.get("type") in ("translation", "collection"):
                    kind = prohibition_check["type"]
                    conf_key = (
                        "translation_of_light" if kind == "translation" else "collection_of_light"
                    )
                    extra = (
                        {"translator": prohibition_check.get("translator")}
                        if kind == "translation"
                        else {"collector": prohibition_check.get("collector")}
                    )
                    return {
                        "perfects": True,
                        "type": kind,
                        "favorable": True,
                        "confidence": getattr(config.confidence.perfection, conf_key),
                        "reason": prohibition_check["reason"],
                        "t_perfect_days": prohibition_check.get("t_event"),
                        "aspect": aspect_type,
                        "mediating_aspect": prohibition_check.get("aspect"),
                        "aspect_quality": prohibition_check.get("quality"),
                        "reception": prohibition_check.get("reception"),
                        **extra,
                        "tags": [{"family": "perfection", "kind": kind}],
                    }

                # Calculate confidence based on aspect quality and timeframe
                base_confidence = config.confidence.perfection.direct_basic

                # Bonus for favorable aspects
                if aspect_type in [Aspect.TRINE, Aspect.SEXTILE]:
                    favorable = True
                elif aspect_type == Aspect.CONJUNCTION:
                    favorable = True  # Generally favorable unless combustion
                else:
                    favorable = False
                    base_confidence -= 10  # Penalty for hard aspects

                # Timing bonus - sooner is better
                if days_to_perfection <= 7:
                    base_confidence += 5
                elif days_to_perfection <= 30:
                    base_confidence += 2

                if mutual != "none":
                    reception_bonus = getattr(
                        config.confidence.reception, f"{mutual}_bonus", 5
                    )
                    base_confidence = min(base_confidence + reception_bonus, 100)
                elif one_way:
                    reception_bonus = getattr(
                        config.confidence.reception, "one_way_bonus", 3
                    )
                    base_confidence = min(base_confidence + reception_bonus, 100)

                return {
                    "perfects": True,
                    "type": "direct_timed",
                    "favorable": favorable,
                    "confidence": base_confidence,
                    "reason": f"Future {aspect_type.display_name} between {querent.value} and {quesited.value} in {days_to_perfection:.1f} days",
                    "t_perfect_days": days_to_perfection,
                    "aspect": aspect_type,
                    "reception": reception_info["type"],
                    "tags": [{"family": "perfection", "kind": "direct_timed"}],
                }

        if alignment_info:
            return alignment_info

        return {"perfects": False, "reason": "No future perfection within timeframe"}
    
    
    def _check_house_placement_perfection(self, chart: HoraryChart, querent: Planet, quesited: Planet, window_days: int = None) -> Dict[str, Any]:
        """Check for perfection via planets in quesited house aspecting house ruler"""
        
        config = cfg()
        
        # Get quesited house information from chart metadata  
        significator_info = getattr(chart, 'significator_info', {})
        quesited_house = significator_info.get('quesited_house')
        
        # Only apply this logic when we have explicit quesited house information
        # (i.e., for career questions where quesited = 10th house)
        if not quesited_house:
            return {"perfects": False}
        
        # Get the house ruler
        house_ruler = chart.house_rulers.get(quesited_house)
        if not house_ruler or house_ruler != quesited:
            return {"perfects": False}
        
        # Find planets in the quesited house (excluding the house ruler itself)
        planets_in_house = []
        for planet, pos in chart.planets.items():
            if pos.house == quesited_house and planet != house_ruler:
                planets_in_house.append(planet)
        
        # Check if any planet in the house has applying aspect to house ruler
        for planet_in_house in planets_in_house:
            
            # Check current applying aspect to house ruler
            aspect = self._find_applying_aspect(chart, planet_in_house, house_ruler)
            if aspect:
                planet_pos = chart.planets[planet_in_house]
                
                # Boost confidence if planet has strong dignity in the house
                base_confidence = config.confidence.perfection.direct_basic
                if planet_pos.dignity_score > 3:
                    base_confidence += 10
                
                # Determine if aspect is favorable  
                favorable = aspect["aspect"] in [Aspect.TRINE, Aspect.SEXTILE, Aspect.CONJUNCTION]
                
                return {
                    "perfects": True,
                    "type": "house_placement",
                    "favorable": favorable,
                    "confidence": base_confidence,
                    "reason": f"{planet_in_house.value} in house {quesited_house} {self._format_aspect_for_display(planet_in_house.value, aspect['aspect'], house_ruler.value, True)}",
                    "planet_in_house": planet_in_house,
                    "house_number": quesited_house,
                    "house_ruler": house_ruler,
                    "aspect": aspect,
                    "tags": [{"family": "perfection", "kind": "house_placement"}]
                }
            
            # Check future applying aspect within timeframe
            if window_days is not None:
                future_aspect = self._check_future_aspect_to_ruler(chart, planet_in_house, house_ruler, window_days)
                if future_aspect and future_aspect.get("perfects", False):
                    return future_aspect
        
        return {"perfects": False}
    
    def _check_future_aspect_to_ruler(self, chart: HoraryChart, planet: Planet, ruler: Planet, window_days: int) -> Dict[str, Any]:
        """Check for future applying aspect from planet to house ruler within timeframe"""
        
        config = cfg()
        max_window = min(window_days, getattr(config.timing, "max_window_days", 365))
        
        planet_pos = chart.planets[planet]
        ruler_pos = chart.planets[ruler]
        
        # Calculate future aspect perfection times
        aspect_types = [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.SQUARE, Aspect.TRINE, Aspect.OPPOSITION]
        
        for aspect_type in aspect_types:
            days_to_perfection = self._calculate_future_aspect_time(
                planet_pos, ruler_pos, aspect_type, chart.julian_day, max_window
            )
            
            if days_to_perfection is not None and 0 < days_to_perfection <= max_window:
                # Check for prohibitions before perfection
                prohibition_check = self._check_future_prohibitions(
                    chart, planet, ruler, days_to_perfection
                )
                if prohibition_check.get("prohibited"):
                    return {
                        "perfects": False,
                        "type": prohibition_check.get("type", "prohibition"),
                        "reason": prohibition_check["reason"],
                    }

                if prohibition_check.get("type") in ("translation", "collection"):
                    kind = prohibition_check["type"]
                    conf_key = (
                        "translation_of_light" if kind == "translation" else "collection_of_light"
                    )
                    extra = (
                        {"translator": prohibition_check.get("translator")}
                        if kind == "translation"
                        else {"collector": prohibition_check.get("collector")}
                    )
                    return {
                        "perfects": True,
                        "type": kind,
                        "favorable": True,
                        "confidence": getattr(config.confidence.perfection, conf_key),
                        "reason": prohibition_check["reason"],
                        "t_perfect_days": prohibition_check.get("t_event"),
                        "planet_in_house": planet,
                        "house_ruler": ruler,
                        "aspect": aspect_type,
                        "mediating_aspect": prohibition_check.get("aspect"),
                        "aspect_quality": prohibition_check.get("quality"),
                        "reception": prohibition_check.get("reception"),
                        **extra,
                        "tags": [{"family": "perfection", "kind": kind}],
                    }

                # Calculate confidence
                base_confidence = config.confidence.perfection.direct_basic

                # Boost for strong dignity
                if planet_pos.dignity_score > 3:
                    base_confidence += 10

                # Determine favorability
                favorable = aspect_type in [Aspect.TRINE, Aspect.SEXTILE, Aspect.CONJUNCTION]
                if not favorable:
                    base_confidence -= 10

                # Timing bonus
                if days_to_perfection <= 7:
                    base_confidence += 5
                elif days_to_perfection <= 30:
                    base_confidence += 2

                return {
                    "perfects": True,
                    "type": "future_house_placement",
                    "favorable": favorable,
                    "confidence": base_confidence,
                    "reason": f"Future {aspect_type.display_name}: {planet.value} in house {planet_pos.house} to {ruler.value} in {days_to_perfection:.1f} days",
                    "t_perfect_days": days_to_perfection,
                    "planet_in_house": planet,
                    "house_ruler": ruler,
                    "aspect": aspect_type,
                    "tags": [{"family": "perfection", "kind": "future_house_placement"}]
                }
        
        return {"perfects": False}
    
    def _check_future_prohibitions(self, chart: HoraryChart, querent: Planet, quesited: Planet,
                                  days_ahead: float) -> Dict[str, Any]:
        """Check if future perfection will be modified by intervening aspects.

        This delegates to :func:`check_future_prohibitions` which implements
        classical Lilly/Sahl rules for prohibition, translation and collection
        of light. The helper is passed a reference to this engine's
        ``_calculate_future_aspect_time`` so it can solve timing analytically.
        """

        return check_future_prohibitions(
            chart, querent, quesited, days_ahead, self._calculate_future_aspect_time
        )
    
    def _check_moon_sun_education_perfection(self, chart: HoraryChart, question_analysis: Dict) -> Dict[str, Any]:
        """Check Moon-Sun aspects in education questions (traditional co-significator analysis)"""
        
        # Moon is always co-significator of querent
        # Sun often represents authority/examiner in education contexts
        moon_aspect = self._find_applying_aspect(chart, Planet.MOON, Planet.SUN)
        
        if moon_aspect:
            # Check if it's a beneficial aspect
            favorable_aspects = ["Conjunction", "Sextile", "Trine"]
            is_favorable = moon_aspect["aspect"].display_name in favorable_aspects
            
            if is_favorable:
                return {
                    "perfects": True,
                    "type": "moon_sun_education",
                    "favorable": True,
                    "confidence": 75,  # Good confidence for traditional co-significator analysis
                    "reason": f"Moon (co-significator) applying {moon_aspect['aspect'].display_name} to Sun (examiner/authority)",
                    "aspect": moon_aspect
                }
        
        # Also check separating aspects (recent perfection can be relevant)
        separating_aspect = self._find_separating_aspect(chart, Planet.MOON, Planet.SUN)
        if separating_aspect:
            favorable_aspects = ["Conjunction", "Sextile", "Trine"]
            is_favorable = separating_aspect["aspect"].display_name in favorable_aspects
            
            if is_favorable:
                return {
                    "perfects": True,
                    "type": "moon_sun_education",
                    "favorable": True,
                    "confidence": 65,  # Slightly lower for separating aspects
                    "reason": f"Moon (co-significator) recently separated from {separating_aspect['aspect'].display_name} to Sun (examiner/authority)",
                    "aspect": separating_aspect
                }
        
        return {
            "perfects": False,
            "reason": "No beneficial Moon-Sun aspects found"
        }
    
    def _find_separating_aspect(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> Optional[Dict]:
        """Find separating aspect between two planets"""
        for aspect in chart.aspects:
            if ((aspect.planet1 == planet1 and aspect.planet2 == planet2) or
                (aspect.planet1 == planet2 and aspect.planet2 == planet1)):
                if not aspect.applying:  # Separating
                    return {
                        "aspect": aspect.aspect,
                        "orb": aspect.orb,
                        "applying": False,
                        "degrees_to_exact": aspect.degrees_to_exact
                    }
        return None
    
    def _check_enhanced_collection_of_light(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Traditional collection of light following Lilly's rules"""
        
        config = cfg()

        candidates: List[Dict[str, Any]] = []

        for planet, pos in chart.planets.items():
            if planet in [querent, quesited]:
                continue
            
            # TRADITIONAL REQUIREMENT 1: Collector must be slower/heavier than both significators
            querent_pos = chart.planets[querent]
            quesited_pos = chart.planets[quesited]
            
            if not (abs(pos.speed) < abs(querent_pos.speed) and abs(pos.speed) < abs(quesited_pos.speed)):
                continue  # Skip if not slower than both
            
            # TRADITIONAL REQUIREMENT 2: Both significators must apply to collector
            aspects_from_querent = self._find_applying_aspect(chart, querent, planet)
            aspects_from_quesited = self._find_applying_aspect(chart, quesited, planet)
            
            if not (aspects_from_querent and aspects_from_quesited):
                continue  # Must have applying aspects from BOTH significators
            
            # TRADITIONAL REQUIREMENT 3: Reception - both significators receive collector in dignities
            # Lilly: "they both receive him in some of their essential dignities"
            querent_receives_collector = self._check_dignified_reception(chart, querent, planet)
            quesited_receives_collector = self._check_dignified_reception(chart, quesited, planet)
            
            # Traditional requirement: BOTH must receive the collector
            if not (querent_receives_collector and quesited_receives_collector):
                continue
            
            # TRADITIONAL REQUIREMENT 4: Timing validation - collection must complete in current signs
            querent_days_to_sign = self._days_to_sign_exit(querent_pos)
            quesited_days_to_sign = self._days_to_sign_exit(quesited_pos)
            
            # Calculate when collection aspects will perfect
            querent_collection_days = self._days_to_aspect_perfection(querent_pos, pos, aspects_from_querent)
            quesited_collection_days = self._days_to_aspect_perfection(quesited_pos, pos, aspects_from_quesited)
            
            # Check timing validity
            timing_valid = True
            if querent_days_to_sign and querent_collection_days > querent_days_to_sign:
                timing_valid = False
            if quesited_days_to_sign and quesited_collection_days > quesited_days_to_sign:
                timing_valid = False

            if not timing_valid:
                continue
            
            # Assess collector's condition and dignity
            collector_strength = pos.dignity_score
            base_confidence = 60
            challenge_reasons = []  # Track challenges, not failures

            # Strong collector increases confidence
            if collector_strength >= 3:
                base_confidence += 15
            elif collector_strength >= 0:
                base_confidence += 5
            else:
                base_confidence -= 10  # Weak collector reduces confidence
                challenge_reasons.append("weak collector")

            # Check if collector is free from major afflictions
            if hasattr(pos, 'solar_condition') and pos.solar_condition.condition == "Combustion":
                base_confidence -= 20  # Combust collector less reliable - challenge, not failure
                challenge_reasons.append("collector combust")

            # Aspect quality influences outcome
            hard_aspect = (
                aspects_from_querent["aspect"] in [Aspect.SQUARE, Aspect.OPPOSITION]
                or aspects_from_quesited["aspect"] in [Aspect.SQUARE, Aspect.OPPOSITION]
            )
            if hard_aspect:
                base_confidence -= 15  # Reduce confidence due to challenges
                challenge_reasons.append("hard aspect")

            favorable = collector_strength >= 0 and not hard_aspect

            days_to_perfection = max(querent_collection_days, quesited_collection_days)

            candidates.append(
                {
                    "collector": planet,
                    "favorable": favorable,
                    "confidence": min(90, max(30, base_confidence)),
                    "strength": collector_strength,
                    "timing_valid": True,
                    "reception": "both_receive_collector",
                    "challenge_reasons": challenge_reasons,
                    "negative_reasons": [],
                    "days_to_perfection": days_to_perfection,
                }
            )

        if not candidates:
            return {"found": False, "candidates": []}

        candidates.sort(key=lambda c: (-c["strength"], c["days_to_perfection"]))
        best = candidates[0]
        result = dict(best)
        result.update({"found": True, "candidates": candidates})
        return result

    def _evaluate_blockers(self, chart: HoraryChart, querent: Planet, quesited: Planet,
                           moon_next_aspect: Dict[str, Any], solar_factors: Dict[str, Any],
                           ignore_void_moon: bool = False, ignore_combustion: bool = False) -> Dict[str, Any]:
        """Collect potential blockers and rank them by severity."""
        blockers: List[Dict[str, Any]] = []
        severity_rank = {"fatal": 0, "severe": 1, "warning": 2}

        # Traditional frustration takes top priority
        frustration = self._check_frustration(chart, querent, quesited)
        if frustration.get("found"):
            blockers.append({
                "type": "frustration",
                "severity": "fatal",
                "reason": frustration["reason"],
                "confidence": frustration.get("confidence", 80),
            })

        # Moon's next aspect denial
        if moon_next_aspect.get("result") == "NO":
            blockers.append({
                "type": "moon_next_aspect",
                "severity": "fatal",
                "reason": f"Moon's next aspect denies perfection: {moon_next_aspect['reason']}",
                "confidence": moon_next_aspect.get("confidence", 75),
            })

        if not ignore_void_moon:
            void_check = self._is_moon_void_of_course_enhanced(chart)
            if void_check["void"] and not void_check.get("exception"):
                blockers.append({
                    "type": "void_of_course",
                    "severity": "warning",
                    "reason": f"Moon void of course: {void_check['reason']}",
                })

        # Significant solar impediments to significators
        if not ignore_combustion:
            for analysis in solar_factors.get("detailed_analyses", {}).values():
                if analysis.get("planet") in [querent.value, quesited.value] and analysis.get("condition") in ["Combustion", "Under the Beams"]:
                    blockers.append({
                        "type": "solar_impediment",
                        "severity": "severe",
                        "reason": f"{analysis['planet']} {analysis['condition'].lower()}",
                    })

        blockers.sort(key=lambda b: severity_rank.get(b.get("severity", "warning"), 99))
        return {"blockers": blockers, "fatal": any(b["severity"] == "fatal" for b in blockers)}

    def _calculate_hybrid_confidence(self, verdict: str, token_score: float, blockers: List[Dict], base_confidence: float = 70) -> Dict[str, Any]:
        """Calculate confidence IN THE GIVEN VERDICT using verdict-first logic.
        
        New Approach - Verdict-First Confidence:
        1. Engine determines verdict (YES/NO) first based on perfection/blockers
        2. Start with base confidence IN THAT VERDICT 
        3. Each factor either increases or decreases confidence in the verdict
        4. Result: "X% confident the answer is [verdict]"
        
        Args:
            verdict: The determined verdict ("YES" or "NO")
            token_score: Net testimony score (positive=YES bias, negative=NO bias)
            blockers: List of factors that may support or oppose the verdict
            base_confidence: Starting confidence in the verdict (default 70)
            
        Returns:
            Dictionary with confidence value and reasoning breakdown
        """
        
        # Start with base confidence in the determined verdict
        current_confidence = base_confidence
        confidence_adjustments = []
        
        # Step 1: Token score alignment with verdict
        if verdict == "YES":
            if token_score > 0:
                # Positive tokens support YES verdict - increase confidence
                boost = min(token_score * 5, 20)  # Max +20% boost
                current_confidence = min(current_confidence + boost, 95)
                confidence_adjustments.append({
                    "factor": f"Positive testimony (score +{token_score})",
                    "impact": f"+{boost}%",
                    "reasoning": "Tokens support YES verdict"
                })
            elif token_score < 0:
                # Negative tokens oppose YES verdict - decrease confidence
                penalty = min(abs(token_score) * 5, 25)  # Max -25% penalty
                # Respect minimum floor for valid perfection cases
                min_floor = 30 if hasattr(self, '_has_valid_perfection') and self._has_valid_perfection else 5
                current_confidence = max(current_confidence - penalty, min_floor)
                confidence_adjustments.append({
                    "factor": f"Negative testimony (score {token_score})",
                    "impact": f"-{penalty}%",
                    "reasoning": "Tokens oppose YES verdict"
                })
        else:  # verdict == "NO"
            if token_score < 0:
                # Negative tokens support NO verdict - increase confidence
                boost = min(abs(token_score) * 5, 20)  # Max +20% boost
                current_confidence = min(current_confidence + boost, 95)
                confidence_adjustments.append({
                    "factor": f"Negative testimony (score {token_score})",
                    "impact": f"+{boost}%",
                    "reasoning": "Tokens support NO verdict"
                })
            elif token_score > 0:
                # Positive tokens oppose NO verdict - decrease confidence
                penalty = min(token_score * 5, 25)  # Max -25% penalty
                current_confidence = max(current_confidence - penalty, 5)
                confidence_adjustments.append({
                    "factor": f"Positive testimony (score +{token_score})",
                    "impact": f"-{penalty}%",
                    "reasoning": "Tokens oppose NO verdict"
                })
        
        # Step 2: Apply factor effects based on whether they support the verdict
        for blocker in blockers:
            blocker_type = blocker.get("type", "unknown")
            blocker_reason = blocker.get("reason", "Unknown factor")
            severity = blocker.get("severity", "warning")
            
            # Determine if this factor supports or opposes the verdict
            supports_verdict = self._factor_supports_verdict(blocker_type, verdict)
            
            if supports_verdict:
                # Factor supports the verdict - increase confidence
                if severity == "fatal":
                    boost = 15
                elif severity == "severe":
                    boost = 10
                else:
                    boost = 5
                
                old_confidence = current_confidence
                current_confidence = min(current_confidence + boost, 95)
                actual_boost = current_confidence - old_confidence
                
                confidence_adjustments.append({
                    "factor": blocker_reason,
                    "impact": f"+{actual_boost}%",
                    "reasoning": f"Supports {verdict} verdict"
                })
            else:
                # Factor opposes the verdict - decrease confidence
                if severity == "fatal":
                    penalty = 20
                elif severity == "severe":
                    penalty = 15
                else:
                    penalty = 8
                
                old_confidence = current_confidence
                # Respect minimum floor for valid perfection cases  
                min_floor = 30 if hasattr(self, '_has_valid_perfection') and self._has_valid_perfection else 5
                current_confidence = max(current_confidence - penalty, min_floor)
                actual_penalty = old_confidence - current_confidence
                
                confidence_adjustments.append({
                    "factor": blocker_reason,
                    "impact": f"-{actual_penalty}%",
                    "reasoning": f"Opposes {verdict} verdict"
                })
        
        # Step 3: Final confidence (5% to 95% range)
        final_confidence = max(min(round(current_confidence), 95), 5)
        
        # CRITICAL DEBUG: Log the complete hybrid calculation
        print(f"  HYBRID CALCULATION BREAKDOWN:")
        print(f"    Start: {base_confidence}% confidence in '{verdict}' verdict")
        print(f"    Token Score: {token_score}")
        print(f"    Current After Tokens: {current_confidence}")
        for adj in confidence_adjustments:
            print(f"    {adj['factor']}: {adj['impact']} - {adj['reasoning']}")
        print(f"    Final: {final_confidence}%")
        
        return {
            "confidence": final_confidence,
            "breakdown": {
                "verdict": verdict,
                "base_confidence": base_confidence,
                "token_score": token_score,
                "confidence_adjustments": confidence_adjustments,
                "final_confidence": final_confidence
            },
            "reasoning": f"{final_confidence}% confident the answer is {verdict}"
        }
        
    def _factor_supports_verdict(self, factor_type: str, verdict: str) -> bool:
        """Determine if a factor supports or opposes the given verdict.
        
        Args:
            factor_type: Type of factor (e.g., "no_perfection", "frustration")
            verdict: The verdict ("YES" or "NO")
            
        Returns:
            True if factor supports verdict, False if it opposes
        """
        # Factors that traditionally support NO verdicts
        no_supporting_factors = {
            "no_perfection", "frustration", "refranation", "abscission",
            "debilitated_significator", "cadent_significator", "void_of_course",
            "solar_impediment", "retrograde_quesited", "saturn_7th"
        }
        
        # Factors that traditionally support YES verdicts  
        yes_supporting_factors = {
            "direct_perfection", "translation_of_light", "collection_of_light",
            "mutual_reception", "exalted_significator", "angular_significator",
            "benefic_testimony", "moon_testimony"
        }
        
        if verdict == "NO":
            return factor_type in no_supporting_factors
        else:  # verdict == "YES"
            return factor_type in yes_supporting_factors

    def _check_frustration(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Traditional frustration: other aspect completes before significators perfect"""
        
        config = cfg()
        
        # TRADITIONAL REQUIREMENT 1: There must be a pending perfection between significators
        direct_aspect = self._find_applying_aspect(chart, querent, quesited)
        if not direct_aspect:
            return {"found": False}  # No pending perfection = no frustration possible

        # Calculate timing for the main perfection
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]
        main_perfection_days = self._days_to_aspect_perfection(querent_pos, quesited_pos, direct_aspect)

        # TRADITIONAL REQUIREMENT 2: Check if any third planet completes aspect first
        for aspect in chart.aspects:
            if not aspect.applying:
                continue  # Only applying aspects can frustrate

            # Identify the frustrating planet and target significator
            frustrating_planet = None
            target_significator = None

            if aspect.planet1 in [querent, quesited] and aspect.planet2 not in [querent, quesited]:
                target_significator = aspect.planet1
                frustrating_planet = aspect.planet2
            elif aspect.planet2 in [querent, quesited] and aspect.planet1 not in [querent, quesited]:
                target_significator = aspect.planet2
                frustrating_planet = aspect.planet1
            else:
                continue  # Not a frustration scenario

            # TRADITIONAL REQUIREMENT 3: Frustrating aspect must complete before significator perfection
            frustrating_pos = chart.planets[frustrating_planet]
            target_pos = chart.planets[target_significator]

            # Compute analytic future time to this specific aspect
            frustrating_days = self._calculate_future_aspect_time(
                target_pos,
                frustrating_pos,
                aspect.aspect,
                chart.julian_day,
                cfg().timing.max_future_days,
            )
            if frustrating_days is None:
                continue

            # In-sign guard for the frustrating contact
            exit_target = days_to_sign_exit(target_pos.longitude, target_pos.speed)
            exit_frustrator = days_to_sign_exit(frustrating_pos.longitude, frustrating_pos.speed)
            if (exit_target is not None and frustrating_days >= exit_target) or (
                exit_frustrator is not None and frustrating_days >= exit_frustrator
            ):
                continue

            # Refranation guard: if either stations before the contact, ignore
            planet_id_target = self.calculator.planets_swe.get(target_pos.planet)
            planet_id_frustrator = self.calculator.planets_swe.get(frustrating_pos.planet)
            try:
                if planet_id_target is not None:
                    st_target = calculate_next_station_time(planet_id_target, chart.julian_day)
                else:
                    st_target = None
            except Exception:
                st_target = None
            try:
                if planet_id_frustrator is not None:
                    st_frustrator = calculate_next_station_time(planet_id_frustrator, chart.julian_day)
                else:
                    st_frustrator = None
            except Exception:
                st_frustrator = None
            if (st_target and (st_target - chart.julian_day) < frustrating_days) or (
                st_frustrator and (st_frustrator - chart.julian_day) < frustrating_days
            ):
                continue

            if frustrating_days < main_perfection_days:

                # Assess severity based on frustrating planet
                base_confidence = config.confidence.denial.frustration
                frustration_type = "general"

                if frustrating_planet == Planet.SATURN:
                    base_confidence += 10  # Saturn frustration more severe
                    frustration_type = "Saturn"
                elif frustrating_planet == Planet.MARS:
                    base_confidence += 5   # Mars frustration significant
                    frustration_type = "Mars"

                # Check reception with frustrating planet (can soften)
                reception_with_frustrator = self._check_dignified_reception(chart, target_significator, frustrating_planet)
                if reception_with_frustrator:
                    base_confidence -= 15  # Reception can redirect rather than deny
                    frustration_type += " with reception"

                return {
                    "found": True,
                    "confidence": min(85, base_confidence),
                    "reason": f"{frustrating_planet.value} aspects {target_significator.value} before significator perfection",
                    "frustrating_planet": frustrating_planet,
                    "target_significator": target_significator,
                    "reception": reception_with_frustrator,
                    "type": frustration_type
                }
        
        return {"found": False}
    
    def _days_to_sign_exit(self, pos: PlanetPosition) -> float:
        """Calculate days until planet exits current sign"""
        try:
            from .calculation.helpers import days_to_sign_exit
            return days_to_sign_exit(pos.longitude, pos.speed)
        except ImportError:
            # Fallback calculation
            degrees_in_sign = pos.longitude % 30
            degrees_remaining = 30 - degrees_in_sign if pos.speed > 0 else degrees_in_sign
            return degrees_remaining / abs(pos.speed) if pos.speed != 0 else None
    
    def _check_moon_translation_pattern(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Check for Moon translation using chart moon_last_aspect and moon_next_aspect data"""
        
        moon_last = getattr(chart, 'moon_last_aspect', None)
        moon_next = getattr(chart, 'moon_next_aspect', None)
        
        if not (moon_last and moon_next):
            return {"found": False}
        
        # Check if Moon separated from one significator and applies to the other
        last_planet_name = getattr(moon_last, 'planet', None)
        next_planet_name = getattr(moon_next, 'planet', None)
        
        if not (last_planet_name and next_planet_name):
            return {"found": False}
        
        # Convert string planet names to Planet enums for comparison
        try:
            last_planet = Planet(last_planet_name.upper())
            next_planet = Planet(next_planet_name.upper()) 
        except (ValueError, AttributeError):
            return {"found": False}
        
        # Check for translation pattern: separated from one significator, applies to the other
        translation_found = False
        sequence = ""
        
        if (last_planet == querent and next_planet == quesited and 
            not getattr(moon_last, 'applying', True) and getattr(moon_next, 'applying', False)):
            translation_found = True
            # CRITICAL FIX: aspect is a string, not an object with display_name
            last_aspect_name = getattr(moon_last, 'aspect', 'aspect')
            next_aspect_name = getattr(moon_next, 'aspect', 'aspect')
            sequence = f"separated from {querent.value} by {last_aspect_name.lower()}, applies to {quesited.value} by {next_aspect_name.lower()}"
            
        elif (last_planet == quesited and next_planet == querent and 
              not getattr(moon_last, 'applying', True) and getattr(moon_next, 'applying', False)):
            translation_found = True  
            # CRITICAL FIX: aspect is a string, not an object with display_name
            last_aspect_name = getattr(moon_last, 'aspect', 'aspect')
            next_aspect_name = getattr(moon_next, 'aspect', 'aspect')
            sequence = f"separated from {quesited.value} by {last_aspect_name.lower()}, applies to {querent.value} by {next_aspect_name.lower()}"
        
        if not translation_found:
            return {"found": False}
            
        # Check for mutual reception to boost confidence
        reception_bonus = 0
        try:
            # Use centralized reception check; if the target receives the Moon, boost confidence
            if self._check_dignified_reception(chart, next_planet, Planet.MOON):
                reception_bonus = 10
        except Exception:
            reception_bonus = 0
        
        next_perfection_days = getattr(moon_next, 'perfection_eta_days', 0) or 0
        
        return {
            "found": True,
            "translator": Planet.MOON,
            "sequence": sequence,
            "favorable": True,  # Translation generally favorable
            "confidence": min(75 + reception_bonus, 85),  # Good confidence for translation
            "timing_days": next_perfection_days,
            "reception_bonus": reception_bonus,
            "challenge_reasons": [],
            "perfection_type": "translation"
        }
    
    def _calculate_enhanced_malefic_penalty(self, chart: HoraryChart, significator: Planet, 
                                           malefic: Planet, aspect, aspect_timing: float = None,
                                           perfection_timing: float = None, post_perfection: bool = False) -> int:
        """Calculate enhanced malefic penalty based on sect, dignity, house, and timing"""
        
        # Get malefic planet data
        malefic_pos = chart.planets[malefic]
        
        # Base penalty by malefic type
        base_penalty = -12 if malefic == Planet.MARS else -10  # Mars slightly worse than Saturn
        
        # 1. SECT MULTIPLIER (Day chart out-of-sect malefics are worse)
        chart_sect = self._determine_chart_sect(chart)
        sect_multiplier = 1.0
        
        if chart_sect == "diurnal":  # Day chart
            if malefic == Planet.MARS:  # Mars out of sect by day
                sect_multiplier = 1.25  # Mars penalties ×1.25 in day charts
            # Saturn in sect by day (no amplification)
        else:  # Night chart  
            if malefic == Planet.SATURN:  # Saturn out of sect by night
                sect_multiplier = 1.25  # Saturn penalties ×1.25 in night charts
            # Mars in sect by night (no amplification)
            
        # 2. ESSENTIAL DIGNITY AMPLIFICATION
        dignity_penalty = 0
        if malefic_pos.essential_dignity <= -4:  # In fall or detriment
            dignity_penalty = -8
        elif malefic_pos.essential_dignity <= -2:  # Debilitated
            dignity_penalty = -4
        elif malefic_pos.essential_dignity >= 4:  # Exalted or domicile - still harmful but less so
            dignity_penalty = +2  # Slight mitigation
            
        # 3. HOUSE PLACEMENT AMPLIFICATION
        house_penalty = 0
        if malefic_pos.house in [6, 8, 12]:  # Traditional malefic houses
            house_penalty = -6
        elif malefic_pos.house in [1, 7, 10]:  # Angular houses (more powerful)
            house_penalty = -4
            
        # 4. ASPECT TYPE AMPLIFICATION  
        # Handle different aspect input types: Aspect enum, aspect data structure, or string
        if hasattr(aspect, 'aspect') and hasattr(aspect.aspect, 'display_name'):
            # aspect is a data structure with an aspect attribute (e.g., chart aspect)
            aspect_name = aspect.aspect.display_name
        elif hasattr(aspect, 'display_name'):
            # aspect is an Aspect enum directly
            aspect_name = aspect.display_name
        elif isinstance(aspect, str):
            # aspect is already a string
            aspect_name = aspect
        else:
            # fallback - convert to string
            aspect_name = str(aspect)
            
        aspect_penalty = 0
        
        if aspect_name.upper() in ['CONJUNCTION']:
            aspect_penalty = -8  # Direct joining with malefic is worst
        elif aspect_name.upper() in ['SQUARE', 'OPPOSITION']:
            aspect_penalty = -6  # Hard aspects
        elif aspect_name.upper() in ['SEXTILE', 'TRINE']:
            aspect_penalty = -3  # Soft aspects still problematic from malefics
            
        # 5. TIMING/SEQUENCE MODULATION
        timing_modifier = 0
        if post_perfection:
            # Post-perfection corruption gets reduced penalty but still significant
            timing_modifier = +4  # Slightly less severe than prohibition
        elif aspect_timing is not None and perfection_timing is not None:
            # Imminent malefic corruption (within 1 day of perfection) is worse
            if abs(aspect_timing - perfection_timing) <= 1.0:
                timing_modifier = -5  # Imminent corruption
                
        # 6. COMBUSTION AMPLIFICATION (if malefic is combust, it's chaotic)
        combustion_penalty = 0
        solar_condition = getattr(malefic_pos, 'solar_condition', {})
        if isinstance(solar_condition, dict) and solar_condition.get('condition') == 'Combustion':
            combustion_penalty = -4  # Combust malefics are unpredictable and harmful
            
        # Calculate base malefic penalty (before sect amplification)
        base_malefic_penalty = (base_penalty + dignity_penalty + house_penalty + aspect_penalty)
        
        # Apply sect multiplier to malefic penalty
        sect_amplified_penalty = int(base_malefic_penalty * sect_multiplier)
        
        # Add timing and combustion modifiers (not amplified by sect)
        total_penalty = sect_amplified_penalty + timing_modifier + combustion_penalty
        
        # Apply reasonable bounds
        if post_perfection:
            # Post-perfection penalties should be meaningful but not as severe as prohibitions
            total_penalty = max(total_penalty, -25)
            total_penalty = min(total_penalty, -5)
        else:
            # Traditional prohibitions can be very severe
            total_penalty = max(total_penalty, -40)
            total_penalty = min(total_penalty, -8)
            
        return int(total_penalty)
    
    def _determine_chart_sect(self, chart: HoraryChart) -> str:
        """Determine if chart is diurnal (day) or nocturnal (night)"""
        sun_pos = chart.planets[Planet.SUN]
        
        # Sun above horizon (houses 7-12) = day chart
        # Sun below horizon (houses 1-6) = night chart
        if sun_pos.house in [7, 8, 9, 10, 11, 12]:
            return "diurnal"
        else:
            return "nocturnal"
    
    def _separate_occurrence_vs_quality(self, chart: HoraryChart, perfection_result: Dict, 
                                       reasoning: List[Dict]) -> Dict[str, Any]:
        """CRITICAL FIX: Separate 'will it happen?' from 'is it favorable?' judgments"""
        
        # Extract occurrence indicators (perfection routes)
        occurrence_score = 0
        occurrence_reasons = []
        
        # Extract quality indicators (condition assessments)
        quality_score = 0
        quality_reasons = []
        
        # Categorize reasoning by type
        for entry in reasoning:
            # Handle both dict and string entries in reasoning
            if isinstance(entry, dict):
                stage = entry.get("stage", "")
                rule = entry.get("rule", "")
                weight = entry.get("weight", 0)
            elif isinstance(entry, str):
                # String entries are usually rules/descriptions
                stage = "General"
                rule = entry
                weight = 0
            else:
                # Skip unknown entry types
                continue
            
            # OCCURRENCE INDICATORS (Will it happen?)
            if any(keyword in stage.lower() for keyword in ["perfection", "testimony", "moon testimony", "reception"]):
                if "perfection" in stage.lower() or "translation" in rule.lower() or "collection" in rule.lower():
                    occurrence_score += abs(weight) if weight > 0 else 0  # Only positive perfection indicators
                    if weight != 0:
                        occurrence_reasons.append(f"{rule} ({'+' if weight > 0 else ''}{weight})")
            
            # QUALITY INDICATORS (Is it favorable?)
            elif any(keyword in stage.lower() for keyword in ["corruption", "impediment", "prohibition", "cadent", "combustion"]):
                quality_score += weight  # Include negative quality factors
                if weight != 0:
                    quality_reasons.append(f"{rule} ({'+' if weight > 0 else ''}{weight})")
        
        # OCCURRENCE JUDGMENT
        will_happen = False
        occurrence_confidence = 0
        
        if perfection_result.get("perfects", False):
            will_happen = True
            # Base occurrence confidence on perfection strength
            if perfection_result["type"] == "translation":
                occurrence_confidence = 75  # Translation is strong occurrence indicator
            elif perfection_result["type"] == "collection":  
                occurrence_confidence = 70  # Collection is good occurrence indicator
            elif perfection_result["type"] == "direct":
                occurrence_confidence = 80  # Direct perfection is strongest
            else:
                occurrence_confidence = 60  # Other perfection types
        else:
            occurrence_confidence = 20  # Low chance without perfection
        
        # QUALITY JUDGMENT (Only relevant if it will happen)
        is_favorable = None
        quality_confidence = 0
        
        if will_happen:
            # Assess quality based on corrupting factors
            major_corruptions = []
            minor_corruptions = []
            
            for entry in reasoning:
                # Handle both dict and string entries in reasoning
                if isinstance(entry, dict):
                    stage = entry.get("stage", "")
                    rule = entry.get("rule", "")
                    weight = entry.get("weight", 0)
                elif isinstance(entry, str):
                    # String entries are usually rules/descriptions
                    stage = "General"
                    rule = entry
                    weight = 0
                else:
                    # Skip unknown entry types
                    continue
                
                # Major quality corruptions
                if weight <= -10:  # Severe penalties
                    major_corruptions.append(rule)
                elif weight <= -5:  # Moderate penalties
                    minor_corruptions.append(rule)
            
            # Calculate quality assessment
            total_corruption_weight = sum(
                entry.get("weight", 0) for entry in reasoning 
                if isinstance(entry, dict) and entry.get("weight", 0) < 0
            )
            
            if total_corruption_weight <= -20:  # Heavy corruption
                is_favorable = False
                quality_confidence = 85
            elif total_corruption_weight <= -10:  # Moderate corruption
                is_favorable = False  
                quality_confidence = 70
            elif total_corruption_weight <= -5:   # Light corruption
                is_favorable = True
                quality_confidence = 60
            else:  # Minimal corruption
                is_favorable = True
                quality_confidence = 75
                
        return {
            "occurrence": {
                "will_happen": will_happen,
                "confidence": occurrence_confidence,
                "reasons": occurrence_reasons,
                "score": occurrence_score
            },
            "quality": {
                "is_favorable": is_favorable,
                "confidence": quality_confidence if will_happen else None,
                "reasons": quality_reasons,
                "score": quality_score,
                "major_corruptions": major_corruptions if will_happen else None,
                "minor_corruptions": minor_corruptions if will_happen else None
            },
            "traditional_summary": self._format_traditional_judgment(will_happen, is_favorable, 
                                                                   occurrence_confidence, quality_confidence)
        }
    
    def _format_traditional_judgment(self, will_happen: bool, is_favorable: bool = None,
                                   occurrence_confidence: int = 0, quality_confidence: int = 0) -> str:
        """Format traditional horary judgment separating occurrence from quality.

        Updated: Use concise phrasing without confidence percentages or arrow.
        Examples:
          - "It will likely happen, but quality is unfavorable."
          - "It will likely happen, and quality is favorable."
          - "It will likely not happen"
          - "It will likely happen (quality assessment uncertain)"
        """

        # Occurrence judgment (concise)
        if will_happen:
            occurrence_text = "It will likely happen"
        else:
            occurrence_text = "It will likely not happen"

        # Quality judgment (only if it will happen)
        if will_happen and is_favorable is not None:
            connector = "and" if is_favorable else "but"
            quality_text = f", {connector} quality is {'favorable' if is_favorable else 'unfavorable'}."
        elif will_happen:
            quality_text = " (quality assessment uncertain)"
        else:
            quality_text = ""

        return f"{occurrence_text}{quality_text}"

    def _get_earliest_perfection_timing(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> float:
        """Get earliest perfection timing to establish chronology gate for prohibitions"""
        earliest_days = float('inf')
        
        # Check Moon's next aspect (often the earliest)
        moon_next = getattr(chart, 'moon_next_aspect', None)
        if moon_next:
            try:
                next_planet = getattr(moon_next, 'planet', None)
                if next_planet in [querent, quesited] and getattr(moon_next, 'applying', False):
                    perfection_days = getattr(moon_next, 'perfection_eta_days', float('inf'))
                    earliest_days = min(earliest_days, perfection_days)
            except Exception:
                pass
        
        # Check direct aspects between significators
        for aspect in chart.aspects:
            if aspect.applying and aspect.time_to_perfection is not None:
                if ((aspect.planet1 == querent and aspect.planet2 == quesited) or
                    (aspect.planet1 == quesited and aspect.planet2 == querent)):
                    earliest_days = min(earliest_days, aspect.time_to_perfection)
        
        # Return finite value or None if no perfection found
        return earliest_days if earliest_days != float('inf') else None
    
    def _check_simple_collection_pattern(self, chart: HoraryChart, querent: Planet, quesited: Planet) -> Dict[str, Any]:
        """Check for simple collection pattern: both significators apply to same slower planet"""
        
        # Find all planets that both significators apply to
        common_targets = {}
        
        # Get aspects from querent
        querent_applies_to = {}
        for aspect in chart.aspects:
            if aspect.planet1 == querent and aspect.applying:
                target = aspect.planet2
                if target not in [querent, quesited]:  # Exclude significators
                    querent_applies_to[target] = aspect
        
        # Get aspects from quesited  
        quesited_applies_to = {}
        for aspect in chart.aspects:
            if aspect.planet1 == quesited and aspect.applying:
                target = aspect.planet2
                if target not in [querent, quesited]:  # Exclude significators
                    quesited_applies_to[target] = aspect
        
        # Find common targets (collection candidates)
        for target in querent_applies_to:
            if target in quesited_applies_to:
                target_pos = chart.planets[target]
                querent_pos = chart.planets[querent]
                quesited_pos = chart.planets[quesited]
                
                # Check if collector is traditionally slower (optional check)
                is_slower = (abs(target_pos.speed) < abs(querent_pos.speed) and 
                           abs(target_pos.speed) < abs(quesited_pos.speed))
                
                common_targets[target] = {
                    'querent_aspect': querent_applies_to[target],
                    'quesited_aspect': quesited_applies_to[target], 
                    'is_slower': is_slower,
                    'dignity': target_pos.dignity_score
                }
        
        if not common_targets:
            return {"found": False}
        
        # Select best collector (prefer slower, well-dignified planets)
        best_collector = None
        best_score = -100
        
        for target, data in common_targets.items():
            score = 0
            if data['is_slower']:
                score += 10
            if data['dignity'] > 0:
                score += data['dignity']
            if target in [Planet.JUPITER, Planet.VENUS]:  # Benefics
                score += 5
            elif target == Planet.SATURN:  # Saturn collection (traditional)
                score += 3
            
            if score > best_score:
                best_score = score
                best_collector = target
        
        if not best_collector:
            return {"found": False}
            
        collector_data = common_targets[best_collector]
        collector_pos = chart.planets[best_collector]
        
        # Calculate timing (when the last significator reaches the collector)
        querent_timing = collector_data['querent_aspect'].time_to_perfection
        quesited_timing = collector_data['quesited_aspect'].time_to_perfection
        collection_timing = max(querent_timing, quesited_timing) if (querent_timing and quesited_timing) else None
        
        # Determine favorability based on collector
        if best_collector in [Planet.JUPITER, Planet.VENUS]:
            favorable = True
            confidence = 65
        elif best_collector == Planet.SATURN and collector_pos.dignity_score >= 0:
            favorable = True  # Saturn collection can be positive with good dignity
            confidence = 55
        else:
            favorable = False  # Mars or debilitated Saturn
            confidence = 45
            
        return {
            "found": True,
            "collector": best_collector,
            "favorable": favorable,
            "confidence": confidence,
            "timing_days": collection_timing,
            "is_traditional_slower": collector_data['is_slower'],
            "collector_dignity": collector_data['dignity'],
            "challenge_reasons": [] if favorable else ["Malefic or debilitated collector"],
            "perfection_type": "collection"
        }
    
    def _days_to_aspect_perfection(self, pos1: PlanetPosition, pos2: PlanetPosition, aspect_info: Dict) -> float:
        """Calculate days until an applying aspect perfects.

        Falls back to ``inf`` when no aspect is present or perfection is
        outside the configured search window."""

        aspect = aspect_info.get("aspect")
        if not aspect:
            return float("inf")

        t = self._calculate_future_aspect_time(
            pos1,
            pos2,
            aspect,
            jd_start=0.0,
            max_days=cfg().timing.max_future_days,
        )
        return t if t is not None else float("inf")
    
    def _enhanced_perfects_in_sign(self, pos1: PlanetPosition, pos2: PlanetPosition,
                                  aspect_info: Dict, chart: HoraryChart) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Enhanced perfection check with directional awareness.

        Returns tuple (perfects, impediment) where impediment details reason if False."""
        
        # Use enhanced sign exit calculations
        days_to_exit_1 = days_to_sign_exit(pos1.longitude, pos1.speed)
        days_to_exit_2 = days_to_sign_exit(pos2.longitude, pos2.speed)

        # Estimate days until aspect perfects using analytic solver
        days_to_perfect = self._calculate_future_aspect_time(
            pos1,
            pos2,
            aspect_info["aspect"],
            chart.julian_day,
            cfg().timing.max_future_days,
        )
        if days_to_perfect is None:
            return False, {"type": "stalled"}

        # NEW: Check for future stations before perfection
        jd_start = chart.julian_day
        planet_id_1 = self.calculator.planets_swe.get(pos1.planet)
        planet_id_2 = self.calculator.planets_swe.get(pos2.planet)

        if planet_id_1 is not None:
            station_jd_1 = calculate_next_station_time(planet_id_1, jd_start)
            if station_jd_1 and (station_jd_1 - jd_start) < days_to_perfect:
                return False, {"type": "refranation", "planet": pos1.planet}

        if planet_id_2 is not None:
            station_jd_2 = calculate_next_station_time(planet_id_2, jd_start)
            if station_jd_2 and (station_jd_2 - jd_start) < days_to_perfect:
                return False, {"type": "refranation", "planet": pos2.planet}

        # Check if either planet exits sign before perfection
        if days_to_exit_1 and days_to_perfect > days_to_exit_1:
            return False, {"type": "sign_exit", "planet": pos1.planet}
        if days_to_exit_2 and days_to_perfect > days_to_exit_2:
            return False, {"type": "sign_exit", "planet": pos2.planet}

        return True, None

    def _calculate_future_aspect_time(
        self,
        pos1: PlanetPosition,
        pos2: PlanetPosition,
        aspect: Aspect,
        jd_start: float,
        max_days: int = 30,
    ) -> Optional[float]:
        """Analytically solve when two planets perfect an aspect.

        Uses signed velocities so retrograde motion is honoured. Returns the
        smallest positive time ``t`` (in days) such that ``(lon1 - lon2)``
        achieves the aspect's angular distance. ``None`` is returned when the
        perfection lies outside ``max_days``.
        """

        target_angles = {
            Aspect.CONJUNCTION: 0,
            Aspect.SEXTILE: 60,
            Aspect.SQUARE: 90,
            Aspect.TRINE: 120,
            Aspect.OPPOSITION: 180,
        }

        A = target_angles.get(aspect)
        if A is None:
            return None

        lon1 = pos1.longitude
        lon2 = pos2.longitude
        speed1 = pos1.speed
        speed2 = pos2.speed

        theta0 = (lon1 - lon2) % 360.0
        v_rel = speed1 - speed2
        if abs(v_rel) < 1e-6:
            return None

        # Compute delta for both aspect polarities (e.g. 90° and -90° for a square)
        delta1 = (A - theta0 + 180.0) % 360.0 - 180.0
        delta2 = (-A - theta0 + 180.0) % 360.0 - 180.0
        delta = delta1 if abs(delta1) < abs(delta2) else delta2

        t = delta / v_rel
        if t < 0:
            t += 360.0 / abs(v_rel)
        if t <= 0 or t > max_days:
            return None
        return t
    
    def _check_enhanced_mutual_reception(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> str:
        """Enhanced mutual reception check using centralized calculator"""
        reception_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet1, planet2)
        return reception_data["type"]
    
    def _detect_reception_between_planets(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> str:
        """CENTRALIZED reception detection using single source of truth"""
        reception_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet1, planet2)
        return reception_data["type"]
    
    def _get_reception_for_structured_output(self, chart: HoraryChart, planet1: Planet, planet2: Planet) -> Dict[str, Any]:
        """Get complete reception data for structured output - prevents contradictions"""
        reception_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet1, planet2)
        return {
            "mutual": reception_data["mutual"],
            "one_way": reception_data["one_way"],
            "display_text": reception_data["display_text"],
            "strength": reception_data["traditional_strength"],
        }
    
    def _check_dignified_reception(self, chart: HoraryChart, receiving_planet: Planet, received_planet: Planet) -> bool:
        """Check if receiving_planet has dignified reception of received_planet using centralized calculator"""
        # Use the new directional method to eliminate confusion about parameter order
        reception_data = self.reception_calculator.does_planet_receive(chart, receiving_planet, received_planet)
        return reception_data["has_reception"]
    
    
    def _apply_confidence_threshold(self, result: str, confidence: int, reasoning: List[str]) -> tuple:
        """Apply confidence threshold - low confidence YES should not auto-deny"""

        # When confidence is below 50%, treat YES results cautiously
        if confidence < 50 and result == "YES":
            if confidence < 30:
                reasoning.append(
                    f"Very low confidence ({confidence}%) - result inconclusive despite perfection"
                )
                return "INCONCLUSIVE", max(confidence, 20)
            else:
                reasoning.append(
                    f"Low confidence ({confidence}%) - positive indication with caution"
                )
                return "YES", confidence

        return result, confidence
    
    def _classify_question_intent(self, question: str, category: str) -> str:
        """Classify if question asks for OCCURRENCE ('will it happen?') or QUALITY ('is it favorable?')"""
        
        if not question:
            return "OCCURRENCE"  # Default fallback
            
        question_lower = question.lower().strip()
        
        # QUALITY patterns - asking for evaluation/advisability
        quality_indicators = [
            'should i', 'should we', 'is it good', 'is it wise', 'is it favorable',
            'is it advisable', 'is it beneficial', 'is it worth', 'is it right',
            'good idea', 'wise to', 'advisable to', 'beneficial to', 'favorable',
            'worth it', 'good for me', 'good for us', 'right choice', 'best option'
        ]
        
        # OCCURRENCE patterns - asking for events/outcomes
        occurrence_indicators = [
            'will i', 'will we', 'will it', 'will they', 'will he', 'will she',
            'am i going to', 'are we going to', 'is it going to',
            'going to happen', 'will happen', 'going to get', 'will get',
            'will receive', 'will find', 'will succeed', 'will fail',
            'will be', 'going to be', 'happen', 'occur', 'come to pass'
        ]
        
        # Check for QUALITY indicators first (more specific)
        for indicator in quality_indicators:
            if indicator in question_lower:
                return "QUALITY"
                
        # Check for OCCURRENCE indicators
        for indicator in occurrence_indicators:
            if indicator in question_lower:
                return "OCCURRENCE"
        
        # Category-based defaults for ambiguous questions
        quality_categories = ["ADVICE", "CHOICE", "DECISION"]
        occurrence_categories = ["EVENT", "OUTCOME", "SUCCESS", "TRAVEL"]
        
        if category.upper() in quality_categories:
            return "QUALITY"
        elif category.upper() in occurrence_categories:
            return "OCCURRENCE"
            
        # Default to OCCURRENCE for traditional horary questions
        return "OCCURRENCE"
    
    def _check_moon_next_aspect_to_significators(self, chart: HoraryChart, querent: Planet, quesited: Planet, ignore_void_moon: bool = False, question_analysis: Dict = None) -> Dict[str, Any]:
        """Check if Moon's next applying aspect to either significator is decisive (ENHANCED with L10 logic)"""
        
        next_aspect = chart.moon_next_aspect
        
        # No next aspect = not decisive
        if not next_aspect:
            return {"decisive": False}
        
        # Check for L10 ruler in education questions
        l10_ruler = None
        is_education = question_analysis and question_type == Category.EDUCATION
        if is_education:
            l10_ruler = chart.house_rulers.get(10)
        
        # Next aspect must be to one of the significators OR L10 ruler in education
        target_planets = {querent, quesited}
        if l10_ruler:
            target_planets.add(l10_ruler)
            
        if next_aspect.planet not in target_planets:
            return {"decisive": False}
        
        # Check if Moon is void of course (reduces decisiveness but doesn't eliminate)
        void_of_course = False
        if not ignore_void_moon:
            void_check = self._is_moon_void_of_course_enhanced(chart)
            if void_check["void"] and not void_check["exception"]:
                void_of_course = True
        
        # Determine favorability
        favorable_aspects = [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
        favorable = next_aspect.aspect in favorable_aspects
        
        # Calculate base confidence
        base_confidence = 75 if favorable else 65  # Moon aspects are influential

        # When the Moon is applying to a significator, it's a strong positive
        # testimony. Boost confidence substantially but keep it purely
        # additive so it cannot flip a YES into a NO on its own.
        if next_aspect.applying and favorable:
            base_confidence += 20
        
        # ENHANCED: L10 bonus for education questions
        config = cfg()
        l10_bonus_applied = False
        caps = {"favorable": config.confidence.lunar_confidence_caps.favorable}
        
        if l10_ruler and next_aspect.planet == l10_ruler and next_aspect.applying:
            if next_aspect.aspect in [Aspect.TRINE, Aspect.SEXTILE]:
                l10_bonus = getattr(config.confidence.moon, "to_house_10_bonus", 8)
                cap_lift = getattr(config.confidence.moon, "to_house_10_cap_lift", 5)
                base_confidence += l10_bonus
                caps["favorable"] = max(caps["favorable"], 80 + cap_lift)
                l10_bonus_applied = True

        # Void of course reduces confidence but doesn't eliminate decisiveness
        if void_of_course:
            base_confidence -= 15
        
        # Very close aspects (within 1°) are more decisive
        if next_aspect.orb <= 1.0:
            base_confidence += 10
        
        # Traditional rule: Moon's next applying aspect to significator is decisive unless void + unfavorable
        decisive = True
        if void_of_course and not favorable:
            decisive = False  # Void + unfavorable Moon aspect = not decisive
            
        result = "YES" if favorable else "NO"
        
        # Check for reception with target planet (can improve unfavorable aspects)
        reception = self._detect_reception_between_planets(chart, Planet.MOON, next_aspect.planet)
        if reception != "none" and not favorable:
            # Reception can soften unfavorable Moon aspects
            base_confidence += 10
            result = "UNCLEAR"  # Reception creates uncertainty instead of clear NO
        
        return {
            "decisive": decisive,
            "result": result,
            "confidence": base_confidence,
            "reason": f"Moon next {next_aspect.aspect.display_name}s {next_aspect.planet.value} in {next_aspect.perfection_eta_description}",
            "timing": next_aspect.perfection_eta_description,
            "reception": reception,
            "l10_bonus_applied": l10_bonus_applied,
            "caps": caps,
            "void_moon": void_of_course
        }

    def _check_sun_applying_to_10th_ruler(self, chart: HoraryChart) -> Optional[Dict[str, Any]]:
        """Detect if the Sun applies to the 10th house ruler within 3°"""

        tenth_ruler = chart.house_rulers.get(10)
        if not tenth_ruler:
            return None

        sun_pos = chart.planets[Planet.SUN]
        ruler_pos = chart.planets[tenth_ruler]

        separation = abs(sun_pos.longitude - ruler_pos.longitude)
        if separation > 180:
            separation = 360 - separation
        if separation > 3:
            return None

        # Determine if Sun is applying (separation decreasing)
        time_increment = 0.1
        future_sun = (sun_pos.longitude + sun_pos.speed * time_increment) % 360
        future_ruler = (ruler_pos.longitude + ruler_pos.speed * time_increment) % 360
        future_sep = abs(future_sun - future_ruler)
        if future_sep > 180:
            future_sep = 360 - future_sep
        if future_sep >= separation:
            return None

        combust = False
        if hasattr(ruler_pos, "solar_condition") and ruler_pos.solar_condition.condition == "Combustion":
            combust = True

        return {"ruler": tenth_ruler, "combust": combust}
    
    def _format_reception_for_display(self, reception_type: str, planet1: Planet, planet2: Planet, chart: HoraryChart) -> str:
        """Format reception analysis for user-friendly display using centralized calculator"""
        reception_data = self.reception_calculator.calculate_comprehensive_reception(chart, planet1, planet2)
        return reception_data["display_text"]
    
    def _is_aspect_favorable(self, aspect: Aspect, reception: str) -> bool:
        """Determine if aspect is favorable (preserved)"""
        
        favorable_aspects = [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
        unfavorable_aspects = [Aspect.SQUARE, Aspect.OPPOSITION]
        
        base_favorable = aspect in favorable_aspects
        
        # Mutual reception can overcome bad aspects
        if reception in ["mutual_rulership", "mutual_exaltation", "mixed_reception"]:
            return True
        
        return base_favorable
    
    def _is_aspect_favorable_enhanced(self, aspect: Aspect, reception_info: Dict[str, Any], chart: HoraryChart, querent: Planet, quesited: Planet) -> Tuple[bool, List[str], bool]:
        """Enhanced aspect favorability returning penalty reasons for weak/cadent conditions"""

        favorable_aspects = [Aspect.CONJUNCTION, Aspect.SEXTILE, Aspect.TRINE]
        unfavorable_aspects = [Aspect.SQUARE, Aspect.OPPOSITION]

        base_favorable = aspect in favorable_aspects

        reception_type = reception_info.get("type", "none")
        mutual = reception_info.get("mutual", "none")
        one_way = reception_info.get("one_way", [])

        # Mutual reception can overcome bad aspects completely
        if mutual in ["mutual_rulership", "mutual_exaltation", "mixed_reception"]:
            return True, [], bool(one_way)

        # Evaluate cadent/weak conditions for confidence penalties when no mutual reception
        querent_pos = chart.planets[querent]
        quesited_pos = chart.planets[quesited]

        cadent_houses = [3, 6, 9, 12]
        penalty_reasons = []
        if mutual == "none":
            if quesited_pos.house in cadent_houses:
                penalty_reasons.append(f"{quesited.value} in cadent {quesited_pos.house}th house")
            if quesited_pos.dignity_score < -5:
                penalty_reasons.append(f"{quesited.value} severely weak (dignity {quesited_pos.dignity_score})")
            if querent_pos.house in cadent_houses:
                penalty_reasons.append(f"{querent.value} in cadent {querent_pos.house}th house")
            if querent_pos.dignity_score < -5:
                penalty_reasons.append(f"{querent.value} severely weak (dignity {querent_pos.dignity_score})")

        if aspect in unfavorable_aspects:
            return False, penalty_reasons, bool(one_way)

        return base_favorable, penalty_reasons, bool(one_way)
    
    def _analyze_enhanced_solar_factors(self, chart: HoraryChart, querent: Planet, quesited: Planet, 
                                      ignore_combustion: bool = False) -> Dict:
        """Enhanced solar factors analysis with configuration - FIXED serialization"""
        
        solar_analyses = getattr(chart, 'solar_analyses', {})
        
        # Count significant solar conditions
        cazimi_planets = []
        combusted_planets = []
        under_beams_planets = []
        
        for planet, analysis in solar_analyses.items():
            if analysis.condition == SolarCondition.CAZIMI:
                cazimi_planets.append(planet)
            elif analysis.condition == SolarCondition.COMBUSTION and not ignore_combustion:
                combusted_planets.append(planet)
            elif analysis.condition == SolarCondition.UNDER_BEAMS and not ignore_combustion:
                under_beams_planets.append(planet)
        
        # Build summary with override notes
        summary_parts = []
        if cazimi_planets:
            summary_parts.append(f"Cazimi: {', '.join(p.value for p in cazimi_planets)}")
        if combusted_planets:
            summary_parts.append(f"Combusted: {', '.join(p.value for p in combusted_planets)}")
        if under_beams_planets:
            summary_parts.append(f"Under Beams: {', '.join(p.value for p in under_beams_planets)}")
        
        if ignore_combustion and (combusted_planets or under_beams_planets):
            summary_parts.append("(Combustion effects ignored by override)")
        
        # Convert detailed analyses for JSON serialization
        detailed_analyses_serializable = {}
        for planet, analysis in solar_analyses.items():
            detailed_analyses_serializable[planet.value] = {
                "planet": planet.value,
                "distance_from_sun": round(analysis.distance_from_sun, 4),
                "condition": analysis.condition.condition_name,
                "dignity_modifier": analysis.condition.dignity_modifier if not (ignore_combustion and analysis.condition in [SolarCondition.COMBUSTION, SolarCondition.UNDER_BEAMS]) else 0,
                "description": analysis.condition.description,
                "exact_cazimi": bool(analysis.exact_cazimi),
                "traditional_exception": bool(analysis.traditional_exception),
                "effect_ignored": ignore_combustion and analysis.condition in [SolarCondition.COMBUSTION, SolarCondition.UNDER_BEAMS]
            }
        
        return {
            "significant": len(summary_parts) > 0,
            "summary": "; ".join(summary_parts) if summary_parts else "No significant solar conditions",
            "cazimi_count": len(cazimi_planets),
            "combustion_count": len(combusted_planets) if not ignore_combustion else 0,
            "under_beams_count": len(under_beams_planets) if not ignore_combustion else 0,
            "detailed_analyses": detailed_analyses_serializable,
            "combustion_ignored": ignore_combustion
        }
    
    def _check_theft_loss_specific_denials(self, chart: HoraryChart, question_type: Category,
                                         querent_planet: Planet, quesited_planet: Planet) -> List[str]:
        """Check for traditional theft/loss-specific denial factors (ENHANCED)"""
        denial_reasons = []
        
        if question_type != Category.LOST_OBJECT:
            return denial_reasons
        
        config = cfg()
        querent_pos = chart.planets[querent_planet]
        quesited_pos = chart.planets[quesited_planet] 
        moon_pos = chart.planets[Planet.MOON]
        
        # Traditional theft/loss denial factors
        
        # 1. L2 (possessions) severely afflicted and cadent
        if quesited_planet == chart.house_rulers[2]:  # L2 question
            angularity = self.calculator._get_traditional_angularity(quesited_pos.longitude, chart.houses, quesited_pos.house)
            
            if angularity == "cadent" and quesited_pos.dignity_score <= -5:
                denial_reasons.append(f"L2 ({quesited_planet.value}) cadent and severely afflicted (dignity {quesited_pos.dignity_score}) - item likely destroyed/irretrievable")
        
        # 2. Combustion of significators (traditional theft indicator)
        sun_pos = chart.planets[Planet.SUN]
        for planet, description in [(querent_planet, "querent"), (quesited_planet, "quesited")]:
            planet_pos = chart.planets[planet]
            distance = abs(planet_pos.longitude - sun_pos.longitude)
            if distance > 180:
                distance = 360 - distance
                
            if distance <= config.orbs.combustion_orb:
                denial_reasons.append(f"Combustion of {description} significator ({planet.value}) - matter destroyed/hidden")
        
        # 3. Moon void-of-course in traditional theft contexts
        if hasattr(moon_pos, 'void_course') and moon_pos.void_course:
            denial_reasons.append("Moon void-of-course - no recovery possible")
        
        # 4. Saturn in 7th house (traditional "no recovery" indicator)
        saturn_pos = chart.planets[Planet.SATURN]
        if saturn_pos.house == 7:
            denial_reasons.append("Saturn in 7th house - traditional denial of recovery")
        
        # 5. No translation or collection possible (significators too weak)
        if querent_pos.dignity_score <= -8 and quesited_pos.dignity_score <= -8:
            denial_reasons.append("Both significators severely debilitated - no planetary strength for recovery")
        
        # 6. Mars (natural significator of theft) strongly placed but opposing recovery
        mars_pos = chart.planets[Planet.MARS]
        if mars_pos.dignity_score >= 3:  # Well-dignified Mars
            # Check if Mars opposes the significators
            for sig_planet in [querent_planet, quesited_planet]:
                sig_pos = chart.planets[sig_planet]
                aspect_diff = abs(mars_pos.longitude - sig_pos.longitude)
                if aspect_diff > 180:
                    aspect_diff = 360 - aspect_diff
                
                if 172 <= aspect_diff <= 188:  # Opposition within 8° orb
                    denial_reasons.append(f"Well-dignified Mars opposes {sig_planet.value} - theft/loss strongly indicated")
        
        # 7. South Node conjunct significators (traditional loss indicator) 
        # Note: Would need South Node calculation - placeholder for now
        
        return denial_reasons

    def _check_malefic_prohibition(self, chart: HoraryChart, planet: Planet, context_house: int = None, earliest_perfection_days: float = None) -> Dict[str, Any]:
        """Check if a significator applies to a malefic before any beneficial perfection can occur.
        
        This is a general prohibition principle: if a significator applies to Mars or Saturn 
        (especially in houses relevant to the question), it indicates obstruction, damage, or denial.
        """
        
        malefics = {Planet.MARS, Planet.SATURN}
        prohibitions = []
        
        planet_pos = chart.planets[planet]
        
        # Check all aspects from this planet
        for aspect in chart.aspects:
            if aspect.planet1 == planet and aspect.applying:
                target_planet = aspect.planet2
                target_pos = chart.planets[target_planet]
                
                if target_planet in malefics:
                    # Check if prohibition is particularly strong
                    severity = "moderate"
                    reasoning = f"{planet.value} applies {aspect.aspect.display_name.lower()} to {target_planet.value}"
                    
                    # Enhanced severity conditions
                    if aspect.aspect == Aspect.CONJUNCTION:
                        severity = "severe"
                        reasoning += " (conjunction = direct joining with malefic)"
                        
                    if context_house and target_pos.house == context_house:
                        severity = "severe" 
                        reasoning += f" in {self._ordinal(context_house)} house (malefic on relevant ground)"
                        
                    if target_pos.dignity_score >= 3:  # Well-dignified malefic
                        severity = "severe"
                        reasoning += " (well-dignified malefic = strong opposition)"
                        
                    # CHRONOLOGY GATE: Only apply prohibition if it occurs BEFORE earliest perfection
                    aspect_timing = aspect.time_to_perfection if hasattr(aspect, 'time_to_perfection') else None
                    
                    # If we have timing data, check chronology
                    if earliest_perfection_days is not None and aspect_timing is not None:
                        if aspect_timing > earliest_perfection_days:
                            # This malefic aspect occurs AFTER perfection - not a prohibition
                            # Classify as post-perfection complication with reduced penalty
                            # CRITICAL FIX: Enhanced malefic penalty calculation
                            weight = self._calculate_enhanced_malefic_penalty(
                                chart, planet, target_planet, aspect,
                                aspect_timing, earliest_perfection_days,
                                post_perfection=True
                            )
                            reasoning += f" (post-perfection corruption - enhanced penalty)"
                        else:
                            # Traditional prohibition - occurs before perfection
                            weight = self._calculate_enhanced_malefic_penalty(
                                chart, planet, target_planet, aspect,
                                aspect_timing, earliest_perfection_days,
                                post_perfection=False
                            )
                    else:
                        # No timing data available - use enhanced weights
                        weight = self._calculate_enhanced_malefic_penalty(
                            chart, planet, target_planet, aspect,
                            None, None, post_perfection=False
                        )
                    
                    prohibitions.append({
                        "malefic": target_planet,
                        "aspect": aspect.aspect,
                        "severity": severity,
                        "reasoning": reasoning,
                        "weight": weight,
                        "house": target_pos.house,
                        "timing": aspect_timing,
                        "chronology_checked": earliest_perfection_days is not None
                    })
        
        return {
            "has_prohibition": len(prohibitions) > 0,
            "prohibitions": prohibitions,
            "total_weight": sum(p["weight"] for p in prohibitions)
        }

    def _ordinal(self, n):
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def _calculate_pet_safety_score(self, chart: HoraryChart, question_analysis: Dict, 
                                   querent_planet: Planet, quesited_planet: Planet) -> Dict[str, Any]:
        """Pet-specific safety analysis using traditional horary principles"""
        
        if question_analysis.get("question_type") != Category.PET:
            return {"score": 0, "factors": []}
            
        question_intent = question_analysis.get("question_intent", "SAFETY")
        if question_intent != "SAFETY":
            return {"score": 0, "factors": []}
        
        safety_score = 0
        factors = []
        
        # 1. Mutual reception between L6 and any benefic = +9 safety points
        benefics = {Planet.VENUS, Planet.JUPITER}
        
        # Check L6 reception with each benefic individually
        for benefic in benefics:
            if benefic != quesited_planet:  # Don't double-count if L6 is itself a benefic
                reception_info = self._get_reception_for_structured_output(chart, quesited_planet, benefic)
                mutual_type = reception_info.get("mutual", "none")
                
                # DEBUG: Temporary print to see what's happening
                print(f"PET SAFETY DEBUG: L6 {quesited_planet.value} ↔ {benefic.value} reception = {mutual_type}")
                
                if mutual_type in ["mutual_rulership", "mutual_exaltation", "mixed_reception"]:
                    safety_score += 9
                    factors.append(("Mutual reception between L6 ({}) and {}".format(quesited_planet.value, benefic.value), 9))
                    
                    # +2 more if benefic is in 6th house
                    benefic_pos = chart.planets[benefic]
                    if benefic_pos.house == 6:
                        safety_score += 2
                        factors.append(("{} (benefic) in 6th house".format(benefic.value), 2))
                    break  # Only count the first mutual reception found
        
        # If L6 itself is a benefic, check its reception with the other significator
        if quesited_planet in benefics:
            reception_info = self._get_reception_for_structured_output(chart, querent_planet, quesited_planet)
            mutual_type = reception_info.get("mutual", "none")
            
            if mutual_type in ["mutual_rulership", "mutual_exaltation", "mixed_reception"]:
                safety_score += 9
                factors.append(("Mutual reception between significators (L6 benefic)".format(quesited_planet.value), 9))
                
                # +2 more if L6 benefic is in 6th house
                quesited_pos = chart.planets[quesited_planet]
                if quesited_pos.house == 6:
                    safety_score += 2
                    factors.append(("L6 benefic {} in 6th house".format(quesited_planet.value), 2))
        
        # 2. Venus in 6th = +4 safety (protection/help) - only if not already counted in reception
        venus_pos = chart.planets[Planet.VENUS]
        if venus_pos.house == 6:
            # Check if we already counted Venus in the mutual reception logic above
            already_counted_venus_in_6th = any("Venus (benefic) in 6th house" in str(factor) or "L6 benefic Venus in 6th house" in str(factor) for factor, _ in factors)
            if not already_counted_venus_in_6th:
                safety_score += 4
                factors.append(("Venus in 6th house (protection)", 4))
        
        # 3. Mars in 6th - sect-aware analysis
        mars_pos = chart.planets[Planet.MARS]
        if mars_pos.house == 6:
            # Check if Mars is of the sect (night chart benefits Mars)
            sun_pos = chart.planets[Planet.SUN]
            is_day_chart = sun_pos.house in [7, 8, 9, 10, 11, 12]
            
            if not is_day_chart:  # Night chart - Mars is of sect
                safety_score -= 2
                factors.append(("Mars in 6th (night sect - manageable agitation)", -2))
            else:  # Day chart - Mars is contrary to sect
                safety_score -= 4
                factors.append(("Mars in 6th (day sect - stronger malefic influence)", -4))
        
        # 4. End-of-matter analysis: Strong L4/planets in 4th = stable resolution
        fourth_house_ruler = chart.house_rulers.get(4)
        if fourth_house_ruler:
            ruler_pos = chart.planets[fourth_house_ruler]
            
            # Saturn cazimi in 4th = +4 (exact scenario from user's chart)
            if (fourth_house_ruler == Planet.SATURN and 
                hasattr(ruler_pos, 'solar_condition') and 
                ruler_pos.solar_condition.condition == "Cazimi"):
                safety_score += 4
                factors.append(("Saturn cazimi in 4th house (stable resolution)", 4))
            elif ruler_pos.dignity_score >= 5:  # Well-dignified L4
                safety_score += 3
                factors.append(("Well-dignified L4 ({}) in 4th".format(fourth_house_ruler.value), 3))
        
        # 5. Category-aware retrograde semantics: L6 retrograde = +3 (returns/hiding but alive)
        quesited_pos = chart.planets[quesited_planet] 
        if quesited_pos.retrograde:
            safety_score += 3
            factors.append(("L6 ({}) retrograde (retracing/returning)".format(quesited_planet.value), 3))
        
        # 6. Check L6 ↔ L8 hard applications - if absent, +2 (no death testimony)
        eighth_house_ruler = chart.house_rulers.get(8)
        if eighth_house_ruler and quesited_planet != eighth_house_ruler:
            # Check for hard aspects between L6 and L8
            has_hard_aspect = False
            for aspect in chart.aspects:
                if ((aspect.planet1 == quesited_planet and aspect.planet2 == eighth_house_ruler) or
                    (aspect.planet1 == eighth_house_ruler and aspect.planet2 == quesited_planet)):
                    if aspect.aspect in ["Square", "Opposition"] and aspect.applying:
                        has_hard_aspect = True
                        break
            
            if not has_hard_aspect:
                safety_score += 2  
                factors.append(("No hard L6-L8 applications (no death testimony)", 2))
        
        return {"score": safety_score, "factors": factors}

    def _calculate_moon_testimony_score(self, chart: HoraryChart, question_analysis: Dict,
                                      querent_planet: Planet, quesited_planet: Planet) -> Dict[str, Any]:
        """Moon testimony analysis - separations/applications, speed, void status"""
        
        moon_score = 0
        factors = []
        moon_pos = chart.planets[Planet.MOON]
        
        # 1. Moon separates from benefic: +2
        benefics = {Planet.VENUS, Planet.JUPITER}
        for aspect in chart.aspects:
            if aspect.planet1 == Planet.MOON and not aspect.applying:
                other_planet = aspect.planet2
                if other_planet in benefics:
                    moon_score += 2
                    factors.append(("Moon separates from {} (benefic influence)".format(other_planet.value), 2))
            elif aspect.planet2 == Planet.MOON and not aspect.applying:
                other_planet = aspect.planet1  
                if other_planet in benefics:
                    moon_score += 2
                    factors.append(("Moon separates from {} (benefic influence)".format(other_planet.value), 2))
        
        # 2. Moon applies by easy aspect to planet in 6th: +3
        for aspect in chart.aspects:
            if aspect.applying and aspect.aspect in ["Sextile", "Trine"]:
                target_planet = None
                if aspect.planet1 == Planet.MOON:
                    target_planet = aspect.planet2
                elif aspect.planet2 == Planet.MOON:
                    target_planet = aspect.planet1
                
                if target_planet:
                    target_pos = chart.planets[target_planet]
                    if target_pos.house == 6:
                        moon_score += 3
                        factors.append(("Moon applies {} to {} in 6th".format(aspect.aspect.display_name.lower(), target_planet.value), 3))
        
        # 3. Moon swift (speed > 12°/day): +1
        if moon_pos.speed > 12.0:
            moon_score += 1
            factors.append(("Moon swift (speed {:.1f}°/day)".format(moon_pos.speed), 1))
        
        # 4. Not void of course: +1
        void_check = self._is_moon_void_of_course_enhanced(chart)
        if not void_check.get("void"):
            moon_score += 1
            factors.append(("Moon not void of course", 1))
        
        return {"score": moon_score, "factors": factors}
    
    def _audit_explanation_consistency(self, result: Dict[str, Any], chart: HoraryChart) -> Dict[str, Any]:
        """Audit explanation consistency to ensure reasoning matches judgment (ENHANCED)"""
        audit_notes = []
        reasoning_text = " ".join(
            r.get("rule", str(r)) if isinstance(r, dict) else str(r)
            for r in result.get("reasoning", [])
        )
        judgment = result.get("result", "")
        confidence = result.get("confidence", 0)
        
        # 1. Check judgment-confidence consistency
        if judgment == "YES" and confidence < 50:
            audit_notes.append("WARNING: Positive judgment with low confidence - review logic")
        elif judgment == "NO" and confidence < 50:
            audit_notes.append("WARNING: Negative judgment with low confidence - may be uncertain")
        
        # 2. Check significator identification consistency
        if "Significators:" in reasoning_text:
            # Extract significator mentions from reasoning
            if "Saturn (ruler of 1)" in reasoning_text and hasattr(chart, 'house_rulers'):
                actual_l1_ruler = chart.house_rulers.get(1, None)
                if actual_l1_ruler and actual_l1_ruler.value != "Saturn":
                    audit_notes.append(f"INCONSISTENCY: Reasoning claims Saturn ruler of 1st, but actual ruler is {actual_l1_ruler.value}")
        
        # 3. Check perfection type consistency
        if "Translation of light" in reasoning_text:
            # Should have Moon mentioned as translator
            if "Moon" not in reasoning_text:
                audit_notes.append("INCONSISTENCY: Translation claimed but Moon not mentioned as translator")
        
        # 4. Check reception consistency
        if "reception" in reasoning_text.lower():
            # Reception should boost confidence
            if judgment == "YES" and confidence < 60:
                audit_notes.append("WARNING: Reception claimed but confidence seems low for positive perfection")
        
        # 5. Check denial consistency
        if any(marker in reasoning_text for marker in ["Denial:", "denied"]):
            if judgment != "NO":
                audit_notes.append("SEVERE INCONSISTENCY: Denial mentioned but judgment is not NO")
        
        # 6. Check traditional factor mentions
        traditional_factors_mentioned = []
        if "combustion" in reasoning_text.lower():
            traditional_factors_mentioned.append("combustion")
        if "retrograde" in reasoning_text.lower():
            traditional_factors_mentioned.append("retrograde")
        if "void" in reasoning_text.lower():
            traditional_factors_mentioned.append("void_moon")
        if "cadent" in reasoning_text.lower():
            traditional_factors_mentioned.append("cadent")
        
        # 7. Check for missing critical explanations
        if judgment == "NO" and not any(marker in reasoning_text for marker in ["Denial:", "No perfection", "denied"]):
            audit_notes.append("WARNING: Negative judgment lacks clear denial explanation")
        
        # Add audit results to the response
        if audit_notes:
            result["explanation_audit"] = {
                "issues_found": len(audit_notes),
                "audit_notes": audit_notes,
                "traditional_factors_detected": traditional_factors_mentioned
            }
        else:
            result["explanation_audit"] = {
                "issues_found": 0,
                "audit_notes": [],
                "status": "Explanation appears consistent with judgment"
            }
        
        return result
    
    def _check_intervening_aspects(self, chart: HoraryChart, translator: Planet, separating_aspect, applying_aspect) -> List[str]:
        """Check for aspects that intervene between separation and application.

        The translator's future aspects are analysed up to the perfection time of
        ``applying_aspect``. If any aspect to another planet perfects earlier,
        the translation sequence is considered invalid.
        """

        intervening: List[str] = []

        time_limit = getattr(applying_aspect, "time_to_perfection", None)
        if time_limit is None:
            return intervening

        # Determine planets involved in the known aspects
        sep_other = (
            separating_aspect.planet2
            if separating_aspect.planet1 == translator
            else separating_aspect.planet1
        )
        app_other = (
            applying_aspect.planet2
            if applying_aspect.planet1 == translator
            else applying_aspect.planet1
        )

        other_planets = [
            p for p in chart.planets if p not in {translator, sep_other, app_other}
        ]

        future_aspects = self._calculate_future_planet_aspects(
            chart, translator, other_planets, time_limit
        )

        # The Moon requires special handling to capture cross-sign sequences that
        # the generic calculator might miss.  Reuse the lunar-specific routine for
        # each remaining planet and keep whichever aspect perfects soonest.
        if translator == Planet.MOON:
            for target in other_planets:
                cross = self._calculate_future_lunar_aspects(chart, target, app_other)
                aspect = cross.get(target)
                if aspect and (
                    target not in future_aspects
                    or aspect.time_to_perfection < future_aspects[target].time_to_perfection
                ):
                    future_aspects[target] = aspect

        for planet, aspect in future_aspects.items():
            if aspect.time_to_perfection is not None and aspect.time_to_perfection < time_limit:
                aspect_symbol = self._get_aspect_symbol(aspect.aspect.display_name)
                intervening.append(f"{aspect_symbol} to {planet.value}")

        return intervening
    
    def _is_aspect_within_orb_limits(self, chart: HoraryChart, aspect) -> bool:
        """Check if aspect is within proper orb limits using moiety-based calculation"""
        
        # Get planet positions
        planet1 = aspect.planet1
        planet2 = aspect.planet2
        
        pos1 = chart.planets[planet1]
        pos2 = chart.planets[planet2]
        
        # Calculate moiety-based orb limit
        moiety1 = self._get_planet_moiety(planet1)
        moiety2 = self._get_planet_moiety(planet2)
        max_orb = moiety1 + moiety2
        
        # Check if current orb is within the limit
        return aspect.orb <= max_orb
    
    def _get_planet_moiety(self, planet: Planet) -> float:
        """Get traditional moiety for planet"""
        moieties = {
            Planet.SUN: 17.0,
            Planet.MOON: 12.5,
            Planet.MERCURY: 7.0,
            Planet.VENUS: 8.0,
            Planet.MARS: 7.5,
            Planet.JUPITER: 9.0,
            Planet.SATURN: 9.5
        }
        return moieties.get(planet, 8.0)  # Default orb if not found
    
    def _validate_translation_sequence_timing(self, chart: HoraryChart, translator: Planet, 
                                            separating_aspect, applying_aspect) -> bool:
        """Validate that separation occurred before application in translation sequence"""
        
        # For a valid translation sequence:
        # 1. The separating aspect must be past exact (separating = not applying)
        # 2. The applying aspect must be approaching exact (applying = true)
        # 3. The translator must have separated from one planet before applying to the other
        
        if separating_aspect.applying or not applying_aspect.applying:
            return False  # Wrong direction - not proper sequence
        
        # Enhanced timing check: compare degrees to exact
        # The separation should be closer to exact than the application
        # (meaning it happened more recently or will happen sooner)
        if hasattr(separating_aspect, 'degrees_to_exact') and hasattr(applying_aspect, 'degrees_to_exact'):
            # For separating aspect, degrees_to_exact represents how far past exact
            # For applying aspect, degrees_to_exact represents how far to exact

            translation_cfg = getattr(cfg(), "translation", SimpleNamespace())
            max_sep = getattr(translation_cfg, "max_separation_deg", 10.0)
            max_app = getattr(translation_cfg, "max_application_deg", 15.0)

            # Additional validation: ensure the separation is recent enough to be meaningful
            if separating_aspect.degrees_to_exact > max_sep:  # Too far past exact
                return False

            # Ensure application is upcoming (not too far away)
            if applying_aspect.degrees_to_exact > max_app:  # Too far to exact
                return False
        
        return True


# NEW: Top-level HoraryEngine class as required
class HoraryEngine:
    """
    Top-level Horary Engine providing the required judge(question, settings) interface
    This is the main entry point as specified in the requirements
    """
    
    def __init__(self):
        self.engine = EnhancedTraditionalHoraryJudgmentEngine()
    
    def judge(self, question: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for horary judgment as specified in requirements
        
        Args:
            question: The horary question to judge
            settings: Dictionary containing all judgment settings
        
        Returns:
            Dictionary with judgment result and analysis
        """
        logger.info(f"=== JUDGE METHOD CALLED ===")
        logger.info(f"Question: {question}")
        logger.info(f"Settings: {settings}")
        
        # Extract settings with defaults
        location = settings.get("location", "London, England")
        date_str = settings.get("date")
        time_str = settings.get("time") 
        timezone_str = settings.get("timezone")
        use_current_time = settings.get("use_current_time", True)
        manual_houses = settings.get("manual_houses")
        
        # Extract override flags
        ignore_radicality = settings.get("ignore_radicality", False)
        ignore_void_moon = settings.get("ignore_void_moon", False)
        ignore_combustion = settings.get("ignore_combustion", False)
        ignore_saturn_7th = settings.get("ignore_saturn_7th", False)
        
        # Extract reception weighting (now configurable)
        exaltation_confidence_boost = settings.get("exaltation_confidence_boost")
        if exaltation_confidence_boost is None:
            # Use configured default
            exaltation_confidence_boost = cfg().confidence.reception.mutual_exaltation_bonus
        
        # Call the enhanced engine
        logger.info("About to call self.engine.judge_question()...")
        try:
            result = self.engine.judge_question(
                question=question,
                location=location,
                date_str=date_str,
                time_str=time_str,
                timezone_str=timezone_str,
                use_current_time=use_current_time,
                manual_houses=manual_houses,
                ignore_radicality=ignore_radicality,
                ignore_void_moon=ignore_void_moon,
                ignore_combustion=ignore_combustion,
                ignore_saturn_7th=ignore_saturn_7th,
                exaltation_confidence_boost=exaltation_confidence_boost
            )
            logger.info("self.engine.judge_question() completed successfully")
        except Exception as engine_error:
            logger.error(f"ERROR in self.engine.judge_question(): {str(engine_error)}")
            logger.error(f"Exception type: {type(engine_error)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
        
        # ENHANCED: Apply explanation consistency audit
        if hasattr(result, 'get') and result.get('chart_data'):
            chart = result.get('chart_data')  # Chart data for audit
            if chart:
                # Create a simplified chart object for audit
                class AuditChart:
                    def __init__(self, chart_data):
                        self.house_rulers = chart_data.get('house_rulers', {})
                        self.planets = {}
                        # Convert planet data for audit
                        for planet_name, planet_data in chart_data.get('planets', {}).items():
                            class PlanetPos:
                                def __init__(self, data):
                                    self.dignity_score = data.get('dignity_score', 0)
                                    self.house = data.get('house', 1)
                            self.planets[planet_name] = PlanetPos(planet_data)
                        self.houses = chart_data.get('houses', [])
                
                audit_chart = AuditChart(chart)
                result = self.engine._audit_explanation_consistency(result, audit_chart)
        
        return result


# Preserve backward compatibility
TraditionalAstrologicalCalculator = EnhancedTraditionalAstrologicalCalculator
TraditionalHoraryJudgmentEngine = EnhancedTraditionalHoraryJudgmentEngine


# Preserve existing serialization functions with enhancements


# Helper functions for testing and development
def load_test_config(config_path: str) -> None:
    """Load test configuration for unit testing"""
    import os
    from horary_config import HoraryConfig
    
    os.environ['HORARY_CONFIG'] = config_path
    HoraryConfig.reset()


def validate_configuration() -> Dict[str, Any]:
    """Validate current configuration and return status"""
    try:
        config = get_config()
        config.validate_required_keys()
        
        return {
            "valid": True,
            "config_file": os.environ.get('HORARY_CONFIG', 'horary_constants.yaml'),
            "message": "Configuration is valid"
        }
    except HoraryError as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "Configuration validation failed"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "Unexpected error during configuration validation"
        }


def get_configuration_info() -> Dict[str, Any]:
    """Get information about current configuration"""
    try:
        config = get_config()
        
        return {
            "config_file": os.environ.get('HORARY_CONFIG', 'horary_constants.yaml'),
            "timing": {
                "default_moon_speed_fallback": config.get('timing.default_moon_speed_fallback'),
                "max_future_days": config.get('timing.max_future_days')
            },
            "translation": {
                "require_speed_advantage": config.get('translation.require_speed_advantage', True),
                "require_proper_sequence": config.get('translation.require_proper_sequence', False)
            },
            "confidence": {
                "base_confidence": config.get('confidence.base_confidence'),
                "lunar_favorable_cap": config.get('confidence.lunar_confidence_caps.favorable'),
                "lunar_unfavorable_cap": config.get('confidence.lunar_confidence_caps.unfavorable')
            },
            "retrograde": {
                "automatic_denial": config.get('retrograde.automatic_denial', True),
                "dignity_penalty": config.get('retrograde.dignity_penalty', -2)
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to get configuration info"
        }


# Enhanced error handling
class HoraryCalculationError(Exception):
    """Exception raised for calculation errors in horary engine"""
    pass


class HoraryConfigurationError(Exception):
    """Exception raised for configuration errors in horary engine"""
    pass


# Logging setup for the module
def setup_horary_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging for horary engine"""
    import logging
    import sys
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Prevent double logging with root handlers
    logger.propagate = False

    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler with UTF-8 encoding support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # Force UTF-8 encoding on Windows
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass
    logger.addHandler(console_handler)
    
    # File handler if specified with UTF-8 encoding
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Horary engine logging configured at {level} level")


# OVERRIDE METHODS for traditional exceptions to hard denials
class TraditionalOverrides:
    """Helper methods for checking traditional overrides to hard denials"""
    
    @staticmethod
    def check_void_moon_overrides(chart, question_analysis, engine):
        """Check for strong traditional overrides for void Moon denial"""
        
        # Get significators for override checks  
        significators = engine._identify_significators(chart, question_analysis)
        if not significators["valid"]:
            return {"can_override": False}
            
        querent = significators["querent"]
        quesited = significators["quesited"]
        
        # Moon carries light cleanly - strongest override
        moon_translation = TraditionalOverrides.check_moon_translation_clean(chart, querent, quesited)
        if moon_translation["clean"]:
            return {
                "can_override": True,
                "reason": moon_translation["reason"],
                "override_type": "moon_translation"
            }
        
        return {"can_override": False}
    
    @staticmethod
    def check_moon_translation_clean(chart, querent, quesited):
        """Check if Moon translates light cleanly between significators"""
        
        moon_pos = chart.planets[Planet.MOON]
        
        # Find Moon's aspects to both significators
        moon_to_querent = None
        moon_to_quesited = None
        
        for aspect in chart.aspects:
            if ((aspect.planet1 == Planet.MOON and aspect.planet2 == querent) or
                (aspect.planet2 == Planet.MOON and aspect.planet1 == querent)):
                moon_to_querent = aspect
            elif ((aspect.planet1 == Planet.MOON and aspect.planet2 == quesited) or
                  (aspect.planet2 == Planet.MOON and aspect.planet1 == quesited)):
                moon_to_quesited = aspect
        
        # Perfect translation requires applying aspects to both
        if (moon_to_querent and moon_to_quesited and 
            moon_to_querent.applying and moon_to_quesited.applying):
            
            # Check Moon's dignity (well-dignified Moon carries light better)
            if moon_pos.dignity_score >= 0:  # At least neutral dignity
                return {
                    "clean": True,
                    "reason": f"Moon (dignity {moon_pos.dignity_score:+d}) perfectly translates {self._format_aspect_for_display('Moon', moon_to_querent.aspect.display_name, querent.value, moon_to_querent.applying)} then {self._format_aspect_for_display('Moon', moon_to_quesited.aspect.display_name, quesited.value, moon_to_quesited.applying)}"
                }
        
        return {"clean": False}


# Performance monitoring helpers
def profile_calculation(func):
    """Decorator to profile calculation performance"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            
            # Add performance info to result if it's a dict
            if isinstance(result, dict):
                result['_performance'] = {
                    'function': func.__name__,
                    'execution_time_seconds': execution_time
                }
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper


# Module version and compatibility info
__version__ = "2.0.0"
__compatibility__ = {
    "api_version": "1.0",
    "config_version": "1.0",
    "breaking_changes": [],
    "deprecated": []
}


def get_engine_info() -> Dict[str, Any]:
    """Get information about the horary engine"""
    return {
        "version": __version__,
        "compatibility": __compatibility__,
        "configuration_status": validate_configuration(),
        "features": {
            "enhanced_moon_testimony": True,
            "configurable_orbs": True,
            "real_moon_speed": True,
            "enhanced_solar_conditions": True,
            "configurable_void_moon": True,
            "retrograde_penalty_mode": True,
            "translation_without_speed": True,
            "lunar_accidental_dignities": True
        }
    }


# Validate configuration on module import (unless disabled)
if os.environ.get('HORARY_CONFIG_SKIP_VALIDATION') != 'true':
    validation_result = validate_configuration()
    if not validation_result["valid"]:
        logger.warning(f"Configuration validation warning: {validation_result['error']}")
        # Don't raise exception to allow module import - let individual functions handle it
