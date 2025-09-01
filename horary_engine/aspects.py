"""Aspect-related calculations for the horary engine."""

from __future__ import annotations

import datetime
import math
from typing import Dict, List, Optional, Tuple

import swisseph as swe

from horary_config import cfg
try:
    from ..models import Aspect, AspectInfo, LunarAspect, Planet, PlanetPosition
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import Aspect, AspectInfo, LunarAspect, Planet, PlanetPosition
from .calculation.helpers import days_to_sign_exit


def _signed_longitude_delta(lon1: float, lon2: float) -> float:
    """Return signed longitudinal difference lon1-lon2 normalised to [-180, 180]."""

    return (lon1 - lon2 + 180) % 360 - 180


def time_to_perfection(
    pos1: PlanetPosition, pos2: PlanetPosition, aspect: Aspect
) -> float:
    """Return time in days until aspect perfection.

    Positive values mean the planets are applying (perfection in the future),
    negative values mean the aspect perfected in the past (separating).
    Returns ``math.inf`` if there is no relative motion between the planets.
    """

    lambda1 = pos1.longitude
    lambda2 = pos2.longitude
    aspect_deg = aspect.degrees

    # delta is the minimal signed offset in (-180°, 180°]
    delta = ((lambda1 - lambda2 - aspect_deg + 180) % 360) - 180
    v = pos1.speed - pos2.speed
    if v == 0:
        return math.inf

    t = -delta / v
    return t


def _is_orb_shrinking(
    pos1: PlanetPosition, pos2: PlanetPosition, aspect: Aspect, dt_days: float = 0.05
) -> bool:
    """Return True if the orb to the given aspect is decreasing.

    Uses a short forward step in time and compares current vs. future orb.
    This approach naturally respects retrograde motion since it relies on the
    positions implied by each planet's instantaneous speed and direction.
    """

    now_orb = _calculate_orb_to_aspect(pos1, pos2, aspect)
    future_orb = _calculate_orb_to_aspect_at_time(pos1, pos2, aspect, dt_days)
    return future_orb < now_orb


def calculate_moon_last_aspect(
    planets: Dict[Planet, PlanetPosition],
    _jd_ut: float,
) -> Optional[LunarAspect]:
    """Calculate Moon's last separating aspect"""

    moon_pos = planets[Planet.MOON]

    # Look back to find most recent separating aspect
    separating_aspects: List[LunarAspect] = []

    for planet, planet_pos in planets.items():
        if planet == Planet.MOON:
            continue

        # Calculate current separation using signed delta
        separation = abs(_signed_longitude_delta(moon_pos.longitude, planet_pos.longitude))

        # Check each aspect type
        for aspect_type in Aspect:
            orb_diff = abs(separation - aspect_type.degrees)
            max_orb = aspect_type.orb

            # Wider orb for recently separating
            if orb_diff <= max_orb * 1.5:
                t = time_to_perfection(moon_pos, planet_pos, aspect_type)
                if t < 0:
                    degrees_since_exact = orb_diff
                    time_since_exact = -t

                    separating_aspects.append(
                        LunarAspect(
                            planet=planet,
                            aspect=aspect_type,
                            orb=orb_diff,
                            degrees_difference=degrees_since_exact,
                            perfection_eta_days=time_since_exact,
                            perfection_eta_description=f"{time_since_exact:.1f} days ago",
                            applying=False,
                        )
                    )

    # Return most recent (smallest time_since_exact)
    if separating_aspects:
        return min(separating_aspects, key=lambda x: x.perfection_eta_days)

    return None


def calculate_moon_next_aspect(
    planets: Dict[Planet, PlanetPosition],
    _jd_ut: float,
    ignore_orb_for_voc: bool = False,
) -> Optional[LunarAspect]:
    """Calculate Moon's next applying aspect.

    Cross-sign perfection is disallowed; aspects must perfect before the Moon
    changes signs.

    Parameters
    ----------
    ignore_orb_for_voc:
        When True, do not gate candidates by current-orb proximity. This is
        used by the traditional Void-of-Course check, which should consider
        any applying Ptolemaic aspect that perfects before sign exit, even if
        it's currently outside display orbs.
    """

    moon_pos = planets[Planet.MOON]

    # Only consider classical planets as targets
    classical_targets = {
        Planet.SUN,
        Planet.MERCURY,
        Planet.VENUS,
        Planet.MARS,
        Planet.JUPITER,
        Planet.SATURN,
    }

    # Find closest applying aspect
    applying_aspects: List[LunarAspect] = []

    for planet, planet_pos in planets.items():
        if planet not in classical_targets:
            continue

        # Calculate current separation using signed delta
        separation = abs(_signed_longitude_delta(moon_pos.longitude, planet_pos.longitude))

        # Check each aspect type
        for aspect_type in Aspect:
            orb_diff = abs(separation - aspect_type.degrees)
            max_orb = aspect_type.orb

            # For general display/UI we gate by orb. For VOC determination,
            # we must not exclude future applying aspects that are currently
            # outside the orb but still perfect before sign exit.
            if ignore_orb_for_voc or orb_diff <= max_orb:
                t = time_to_perfection(moon_pos, planet_pos, aspect_type)
                if t > 0:
                    within_sign = _will_perfect_before_sign_exit(
                        moon_pos, planet_pos, aspect_type, t
                    )

                    # Skip aspects that perfect after either planet leaves its current sign
                    if not within_sign:
                        continue

                    applying_aspects.append(
                        LunarAspect(
                            planet=planet,
                            aspect=aspect_type,
                            orb=orb_diff,
                            degrees_difference=orb_diff,
                            perfection_eta_days=t,
                            perfection_eta_description=format_timing_description(t),
                            applying=True,
                        )
                    )

    # Return soonest (smallest time_to_exact)
    if applying_aspects:
        return min(applying_aspects, key=lambda x: x.perfection_eta_days)

    return None


def is_moon_separating_from_aspect(
    moon_pos: PlanetPosition,
    planet_pos: PlanetPosition,
    aspect: Aspect,
    jd_ut: float,
) -> bool:
    """Determine if the Moon is separating from the given aspect."""

    t = time_to_perfection(moon_pos, planet_pos, aspect)
    return t < 0 and math.isfinite(t)


def is_moon_applying_to_aspect(
    moon_pos: PlanetPosition,
    planet_pos: PlanetPosition,
    aspect: Aspect,
    jd_ut: float,
) -> bool:
    """Determine if the Moon is applying to the given aspect."""

    # Use kinematic approach for consistency with enhanced aspects
    return _is_orb_shrinking(moon_pos, planet_pos, aspect)


def format_timing_description(days: float) -> str:
    """Format timing description for aspect perfection"""
    if days < 0.5:
        return "Within hours"
    if days < 1:
        return "Within a day"
    if days < 7:
        return f"Within {int(days)} days"
    if days < 30:
        return f"Within {int(days/7)} weeks"
    if days < 365:
        return f"Within {int(days/30)} months"
    return "More than a year"


def calculate_enhanced_aspects(
    planets: Dict[Planet, PlanetPosition], jd_ut: float
) -> List[AspectInfo]:
    """Enhanced aspect calculation with configuration"""
    aspects: List[AspectInfo] = []
    planet_list = list(planets.keys())
    config = cfg()

    for i, planet1 in enumerate(planet_list):
        for planet2 in planet_list[i + 1 :]:
            pos1 = planets[planet1]
            pos2 = planets[planet2]

            # Calculate angular separation using signed delta
            angle_diff = abs(_signed_longitude_delta(pos1.longitude, pos2.longitude))

            # Check each traditional aspect
            aspect_candidates: List[AspectInfo] = []
            for aspect_type in Aspect:
                orb_diff = abs(angle_diff - aspect_type.degrees)

                # ENHANCED: Traditional moiety-based orb calculation
                max_orb = calculate_moiety_based_orb(
                    planet1, planet2, aspect_type, config
                )

                # Fallback to configured orbs if moiety system disabled
                if max_orb == 0:
                    max_orb = aspect_type.orb
                    # Luminary bonuses (legacy)
                    if Planet.SUN in [planet1, planet2]:
                        max_orb += config.orbs.sun_orb_bonus
                    if Planet.MOON in [planet1, planet2]:
                        max_orb += config.orbs.moon_orb_bonus

                if orb_diff <= max_orb:
                    # Classify applying/separating via orb shrink (direction-aware, incl. retrograde)
                    applying_raw = _is_orb_shrinking(pos1, pos2, aspect_type)
                    # Compute analytical time for supplemental timing info only
                    t = time_to_perfection(pos1, pos2, aspect_type)
                    if applying_raw:
                        within_sign = (
                            math.isfinite(t)
                            and t > 0
                            and _will_perfect_before_sign_exit(
                                pos1, pos2, aspect_type, t
                            )
                        )
                    else:
                        within_sign = False

                    # Applying/separating should be determined kinematically
                    # (orb shrinking) and not depend on whether perfection
                    # occurs within the current signs. Cross-sign motion is
                    # indicated separately by `perfection_within_sign`.
                    applying_effective = applying_raw

                    degrees_to_exact, exact_time = calculate_enhanced_degrees_to_exact(
                        pos1, pos2, aspect_type, jd_ut, t
                    )

                    aspect_candidates.append(
                        AspectInfo(
                            planet1=planet1,
                            planet2=planet2,
                            aspect=aspect_type,
                            orb=orb_diff,
                            applying=applying_effective,
                            time_to_perfection=t,
                            perfection_within_sign=within_sign,
                            exact_time=exact_time,
                            degrees_to_exact=degrees_to_exact,
                        )
                    )

            if aspect_candidates:
                # Choose candidate with smallest orb difference
                best_aspect = min(aspect_candidates, key=lambda x: x.orb)
                aspects.append(best_aspect)

    return aspects


def calculate_moiety_based_orb(
    planet1: Planet, planet2: Planet, aspect_type: Aspect, config
) -> float:
    """Calculate traditional moiety-based orb for two planets (ENHANCED)"""

    if not hasattr(config.orbs, "moieties"):
        return 0  # Fallback to legacy system

    # Get planetary full orb values and convert to moieties (half-orbs)
    full_orb1 = getattr(config.orbs.moieties, planet1.value, 0.0)
    full_orb2 = getattr(config.orbs.moieties, planet2.value, 0.0)

    moiety1 = full_orb1 / 2.0
    moiety2 = full_orb2 / 2.0

    # Combined moiety orb (sum of half-orbs)
    combined_moiety = moiety1 + moiety2

    # Traditional aspect-specific adjustments
    if aspect_type in [Aspect.CONJUNCTION, Aspect.OPPOSITION]:
        # Conjunction and opposition get full combined moieties
        return combined_moiety
    if aspect_type in [Aspect.TRINE, Aspect.SQUARE]:
        # Squares and trines get slightly reduced orbs
        return combined_moiety * 0.85
    if aspect_type == Aspect.SEXTILE:
        # Sextiles get more restrictive orbs
        return combined_moiety * 0.7
    return combined_moiety * 0.8  # Other aspects


def is_applying_enhanced(
    pos1: PlanetPosition, pos2: PlanetPosition, aspect: Aspect, jd_ut: float
) -> Tuple[bool, bool, float]:
    """Determine applying status, sign perfection flag and timing.

    Returns a tuple ``(applying, perfection_within_sign, time_to_perfection)``
    where ``applying`` is ``True`` if the orb is shrinking and
    ``perfection_within_sign`` indicates whether the aspect perfects before
    either planet changes signs. ``time_to_perfection`` is the analytical time
    in days until perfection (positive means future, negative means past).
    """

    t = time_to_perfection(pos1, pos2, aspect)
    # Use kinematic approach for applying determination (consistent with enhanced aspects)
    applying = _is_orb_shrinking(pos1, pos2, aspect)
    if applying and math.isfinite(t) and t > 0:
        perfection_within_sign = _will_perfect_before_sign_exit(
            pos1, pos2, aspect, t
        )
    else:
        perfection_within_sign = False

    return applying, perfection_within_sign, t


def _calculate_orb_to_aspect(pos1: PlanetPosition, pos2: PlanetPosition, aspect: Aspect) -> float:
    """Calculate current orb (degrees) to exact aspect"""
    
    # Current angular separation
    separation = abs(_signed_longitude_delta(pos1.longitude, pos2.longitude))
    
    # Distance to exact aspect
    orb_to_exact = abs(separation - aspect.degrees)
    
    # Handle wrap-around (e.g., 358° to 2° is 4°, not 356°)
    if orb_to_exact > 180:
        orb_to_exact = 360 - orb_to_exact
    
    return orb_to_exact


def _calculate_orb_to_aspect_at_time(pos1: PlanetPosition, pos2: PlanetPosition, 
                                   aspect: Aspect, time_days: float) -> float:
    """Calculate orb to exact aspect after specified time"""
    
    # Future positions
    future_pos1_lon = (pos1.longitude + pos1.speed * time_days) % 360
    future_pos2_lon = (pos2.longitude + pos2.speed * time_days) % 360
    
    # Future angular separation
    future_separation = abs(_signed_longitude_delta(future_pos1_lon, future_pos2_lon))
    
    # Future distance to exact aspect
    future_orb = abs(future_separation - aspect.degrees)
    
    # Handle wrap-around
    if future_orb > 180:
        future_orb = 360 - future_orb
    
    return future_orb


def _will_perfect_before_sign_exit(
    pos1: PlanetPosition, pos2: PlanetPosition, aspect: Aspect, t: float
) -> bool:
    """Check if aspect will perfect before either planet exits its current sign"""

    if t <= 0 or not math.isfinite(t):
        return False

    pos1_days_to_exit = days_to_sign_exit(pos1.longitude, pos1.speed)
    pos2_days_to_exit = days_to_sign_exit(pos2.longitude, pos2.speed)

    if pos1_days_to_exit is not None and t > pos1_days_to_exit:
        return False
    if pos2_days_to_exit is not None and t > pos2_days_to_exit:
        return False

    future_pos1_lon = (pos1.longitude + pos1.speed * t) % 360
    future_pos2_lon = (pos2.longitude + pos2.speed * t) % 360

    if int(pos1.longitude // 30) != int(future_pos1_lon // 30):
        return False
    if int(pos2.longitude // 30) != int(future_pos2_lon // 30):
        return False

    return True


def calculate_enhanced_degrees_to_exact(
    pos1: PlanetPosition,
    pos2: PlanetPosition,
    aspect: Aspect,
    jd_ut: float,
    t: Optional[float] = None,
) -> Tuple[float, Optional[datetime.datetime]]:
    """Enhanced degrees and time calculation"""

    # Current separation
    separation = abs(_signed_longitude_delta(pos1.longitude, pos2.longitude))

    # Orb from exact
    orb_from_exact = abs(separation - aspect.degrees)

    # CRITICAL FIX: Use consistent applying determination with kinematic approach
    exact_time = None
    if t is None:
        t = time_to_perfection(pos1, pos2, aspect)
        
    # Use kinematic orb-shrinking approach as single source of truth for applying status
    applying = _is_orb_shrinking(pos1, pos2, aspect)
    
    if applying:
        max_future_days = cfg().timing.max_future_days
        if t < max_future_days:
            try:
                exact_jd = jd_ut + t
                # Convert back to datetime
                year, month, day, hour, minute, second = swe.jdut1_to_utc(
                    exact_jd, 1
                )  # Flag 1 for Gregorian
                exact_time = datetime.datetime(
                    int(year), int(month), int(day), int(hour), int(minute), int(second)
                )
            except Exception:
                exact_time = None
    else:
        # Separating aspects: no future exact_time, time_to_perfection should be negative
        exact_time = None
        if t > 0:  # If somehow positive but not applying, make it negative
            t = -abs(t)

    # If already very close, return small value
    if orb_from_exact < 0.1:
        return 0.1, exact_time

    return orb_from_exact, exact_time
