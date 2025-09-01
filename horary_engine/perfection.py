from __future__ import annotations

from typing import Callable, Dict, Any, List, Optional, Tuple

from horary_config import cfg
from .calculation.helpers import days_to_sign_exit, calculate_next_station_time
from .reception import TraditionalReceptionCalculator
try:
    from ..models import Planet, Aspect, HoraryChart
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import Planet, Aspect, HoraryChart
import swisseph as swe

CLASSICAL_PLANETS: List[Planet] = [
    Planet.SUN,
    Planet.MOON,
    Planet.MERCURY,
    Planet.VENUS,
    Planet.MARS,
    Planet.JUPITER,
    Planet.SATURN,
]

ASPECT_TYPES: List[Aspect] = [
    Aspect.CONJUNCTION,
    Aspect.SEXTILE,
    Aspect.SQUARE,
    Aspect.TRINE,
    Aspect.OPPOSITION,
]


def verb(aspect: Aspect) -> str:
    """Return the verb form for a given aspect."""
    mapping = {
        Aspect.CONJUNCTION: "conjoins",
        Aspect.SEXTILE: "sextiles",
        Aspect.SQUARE: "squares",
        Aspect.TRINE: "trines",
        Aspect.OPPOSITION: "opposes",
    }
    return mapping.get(aspect, f"{aspect.display_name.lower()}s")


def check_future_prohibitions(
    chart: HoraryChart,
    sig1: Planet,
    sig2: Planet,
    days_ahead: float,
    calc_aspect_time: Callable[[Any, Any, Aspect, float, float], float],
) -> Dict[str, Any]:
    """Scan for intervening aspects before a main perfection.

    Parameters
    ----------
    chart : HoraryChart
        Chart containing planetary positions.
    sig1, sig2 : Planet
        Significators forming the main perfection.
    days_ahead : float
        Time until the main perfection in days.
    calc_aspect_time : callable
        Function for computing signed time to an aspect. Positive values are
        future contacts while negative values indicate a recent separation.
    """

    config = cfg()
    require_in_sign = getattr(getattr(config, "perfection", {}), "require_in_sign", True)
    allow_out_of_sign = getattr(getattr(config, "perfection", {}), "allow_out_of_sign", False)

    pos1 = chart.planets[sig1]
    pos2 = chart.planets[sig2]
    reception_calc = TraditionalReceptionCalculator()

    # Local Swiss Ephemeris ID mapping for classical planets
    SWE_ID: Dict[Planet, int] = {
        Planet.SUN: swe.SUN,
        Planet.MOON: swe.MOON,
        Planet.MERCURY: swe.MERCURY,
        Planet.VENUS: swe.VENUS,
        Planet.MARS: swe.MARS,
        Planet.JUPITER: swe.JUPITER,
        Planet.SATURN: swe.SATURN,
    }

    def _leg_valid(p_a, p_b, t_leg: Optional[float]) -> bool:
        """Validate a leg timing with in-sign and station (refranation) checks."""
        if t_leg is None or t_leg <= 0 or t_leg >= days_ahead:
            return False
        if require_in_sign and not allow_out_of_sign:
            exit_a = days_to_sign_exit(p_a.longitude, p_a.speed)
            exit_b = days_to_sign_exit(p_b.longitude, p_b.speed)
            if exit_a is not None and t_leg >= exit_a:
                return False
            if exit_b is not None and t_leg >= exit_b:
                return False
        # CRITICAL FIX: Only apply refranation to primary perfections, NOT prohibitions
        # Prohibition is about one planet interfering with significator perfection
        # If the prohibiting planet will station later, the damage is already done when aspect perfects
        # Traditional refranation only applies when the promising planet itself stations
        
        # Skip refranation check for prohibition aspects - they should trigger regardless
        # The refranation check was incorrectly filtering out valid prohibitions
        
        # Commenting out the refranation check that was causing prohibition failures:
        # jd0 = chart.julian_day
        # for p in (p_a, p_b):
        #     swe_id = SWE_ID.get(p.planet)
        #     if swe_id is None:
        #         continue
        #     try:
        #         st_jd = calculate_next_station_time(swe_id, jd0)
        #     except Exception:
        #         st_jd = None
        #     if st_jd is not None and (st_jd - jd0) < t_leg:
        #         return False
        return True

    def _valid(t: Optional[float], p_a, p_b) -> bool:
        return _leg_valid(p_a, p_b, t)

    events: List[Dict[str, Any]] = []
    earliest_prohibition: Optional[Dict[str, Any]] = None
    earliest_tc_event: Optional[Dict[str, Any]] = None

    def _record_event(event: Dict[str, Any]) -> None:
        nonlocal earliest_prohibition, earliest_tc_event
        events.append(event)
        if event["payload"].get("prohibited"):
            if earliest_prohibition is None or event["t"] < earliest_prohibition["t"]:
                earliest_prohibition = event
        else:
            if earliest_tc_event is None or event["t"] < earliest_tc_event["t"]:
                earliest_tc_event = event

    print(f"\n=== PROHIBITION CHECK DEBUG ===")
    print(f"Checking for prohibitions before {sig1.value}-{sig2.value} perfection in {days_ahead} days")
    
    for planet in CLASSICAL_PLANETS:
        if planet in (sig1, sig2):
            continue
        p_pos = chart.planets.get(planet)
        if not p_pos:
            continue

        print(f"\n--- Checking {planet.value} for prohibition ---")
        for aspect in ASPECT_TYPES:
            t1 = calc_aspect_time(pos1, p_pos, aspect, chart.julian_day, days_ahead)
            t2 = calc_aspect_time(pos2, p_pos, aspect, chart.julian_day, days_ahead)
            
            if planet.value == 'Mars' and aspect.value == 'Square':
                print(f"Mars Square: t1({sig1.value})={t1}, t2({sig2.value})={t2}")

            valid1 = _valid(t1, pos1, p_pos)
            valid2 = _valid(t2, pos2, p_pos)
            
            if planet.value == 'Mars' and aspect.value == 'Square':
                print(f"Mars Square validation: valid1={valid1}, valid2={valid2}")
            if valid1 and valid2:
                if (t1 < 0 < t2) or (t2 < 0 < t1):
                    t_event = t2 if t2 > 0 else t1
                    if abs(p_pos.speed) > max(abs(pos1.speed), abs(pos2.speed)):
                        quality = (
                            "easier"
                            if aspect in (Aspect.CONJUNCTION, Aspect.TRINE, Aspect.SEXTILE)
                            else "with difficulty"
                        )
                        rec1 = reception_calc.calculate_comprehensive_reception(chart, planet, sig1)
                        rec2 = reception_calc.calculate_comprehensive_reception(chart, planet, sig2)
                        has_reception = rec1["type"] != "none" or rec2["type"] != "none"
                        reason = (
                            f"Perfection by translation ({aspect.display_name.lower()}): positive "
                            + (f"({quality})" if quality == "easier" else f"{quality}")
                        )
                        if has_reception and quality == "with difficulty":
                            reason += " (softened by reception)"
                        # Before accepting translation, check for abscission: any other planet
                        # perfects with the receiver before t_event (the later leg).
                        receiver_planet = sig2 if (t2 or 0) >= (t1 or 0) else sig1
                        recv_pos = chart.planets[receiver_planet]
                        for other in CLASSICAL_PLANETS:
                            if other in (planet, sig1, sig2):
                                continue
                            other_pos = chart.planets.get(other)
                            if not other_pos:
                                continue
                            for a2 in ASPECT_TYPES:
                                t_rcv = calc_aspect_time(recv_pos, other_pos, a2, chart.julian_day, days_ahead)
                                if _valid(t_rcv, recv_pos, other_pos) and t_rcv < t_event:
                                    _record_event(
                                        {
                                            "t": t_rcv,
                                            "payload": {
                                                "prohibited": True,
                                                "type": "abscission",
                                                "abscissor": other,
                                                "significator": receiver_planet,
                                                "t_abscission": t_rcv,
                                                "reason": f"{other.value} cuts off light to {receiver_planet.value} before translation",
                                            },
                                        }
                                    )
                        # Check if the translator is cut off before completing translation
                        translator_abscised = False
                        for other in CLASSICAL_PLANETS:
                            if other in (planet, sig1, sig2):
                                continue
                            other_pos = chart.planets.get(other)
                            if not other_pos:
                                continue
                            for a2 in ASPECT_TYPES:
                                t_trans = calc_aspect_time(p_pos, other_pos, a2, chart.julian_day, days_ahead)
                                if _valid(t_trans, p_pos, other_pos) and t_trans < t_event:
                                    _record_event(
                                        {
                                            "t": t_trans,
                                            "payload": {
                                                "prohibited": True,
                                                "type": "abscission",
                                                "abscissor": other,
                                                "significator": planet,
                                                "t_abscission": t_trans,
                                                "reason": f"{other.value} cuts off light carried by {planet.value} before translation",
                                            },
                                        }
                                    )
                                    translator_abscised = True
                                    break
                            if translator_abscised:
                                break

                        if translator_abscised:
                            continue

                        _record_event(
                            {
                                "t": t_event,
                                "payload": {
                                    "prohibited": False,
                                    "type": "translation",
                                    "translator": planet,
                                    "t_event": t_event,
                                    "aspect": aspect,
                                    "quality": quality,
                                    "reception": has_reception,
                                    "reason": reason,
                                },
                            }
                        )
                else:
                    t_first, t_second = (t1, t2) if t1 <= t2 else (t2, t1)
                    if (
                        t_first < t_second
                        and t_first >= 0
                        and abs(p_pos.speed) > max(abs(pos1.speed), abs(pos2.speed))
                    ):
                        t_event = t_second
                        quality = (
                            "easier"
                            if aspect in (Aspect.CONJUNCTION, Aspect.TRINE, Aspect.SEXTILE)
                            else "with difficulty"
                        )
                        rec1 = reception_calc.calculate_comprehensive_reception(chart, planet, sig1)
                        rec2 = reception_calc.calculate_comprehensive_reception(chart, planet, sig2)
                        has_reception = rec1["type"] != "none" or rec2["type"] != "none"
                        reason = (
                            f"Perfection by translation ({aspect.display_name.lower()}): positive "
                            + (f"({quality})" if quality == "easier" else f"{quality}")
                        )
                        if has_reception and quality == "with difficulty":
                            reason += " (softened by reception)"
                        translator_abscised = False
                        for other in CLASSICAL_PLANETS:
                            if other in (planet, sig1, sig2):
                                continue
                            other_pos = chart.planets.get(other)
                            if not other_pos:
                                continue
                            for a2 in ASPECT_TYPES:
                                t_trans = calc_aspect_time(p_pos, other_pos, a2, chart.julian_day, days_ahead)
                                if _valid(t_trans, p_pos, other_pos) and t_trans < t_event:
                                    _record_event(
                                        {
                                            "t": t_trans,
                                            "payload": {
                                                "prohibited": True,
                                                "type": "abscission",
                                                "abscissor": other,
                                                "significator": planet,
                                                "t_abscission": t_trans,
                                                "reason": f"{other.value} cuts off light carried by {planet.value} before translation",
                                            },
                                        }
                                    )
                                    translator_abscised = True
                                    break
                            if translator_abscised:
                                break

                        if translator_abscised:
                            continue

                        _record_event(
                            {
                                "t": t_event,
                                "payload": {
                                    "prohibited": False,
                                    "type": "translation",
                                    "translator": planet,
                                    "t_event": t_event,
                                    "aspect": aspect,
                                    "quality": quality,
                                    "reception": has_reception,
                                    "reason": reason,
                                },
                            }
                        )
                    elif (
                        t_first < t_second
                        and t_first >= 0
                        and abs(p_pos.speed) < min(abs(pos1.speed), abs(pos2.speed))
                    ):
                        t_event = t_second
                        quality = (
                            "easier"
                            if aspect in (Aspect.CONJUNCTION, Aspect.TRINE, Aspect.SEXTILE)
                            else "with difficulty"
                        )
                        rec1 = reception_calc.calculate_comprehensive_reception(chart, planet, sig1)
                        rec2 = reception_calc.calculate_comprehensive_reception(chart, planet, sig2)
                        has_reception = rec1["type"] != "none" or rec2["type"] != "none"
                        reason = (
                            f"Perfection by collection ({aspect.display_name.lower()}): positive "
                            + (f"({quality})" if quality == "easier" else f"{quality}")
                        )
                        if has_reception and quality == "with difficulty":
                            reason += " (softened by reception)"
                        # Abscission guard before collection
                        receiver_planet = sig2 if (t2 or 0) >= (t1 or 0) else sig1
                        recv_pos = chart.planets[receiver_planet]
                        for other in CLASSICAL_PLANETS:
                            if other in (planet, sig1, sig2):
                                continue
                            other_pos = chart.planets.get(other)
                            if not other_pos:
                                continue
                            for a2 in ASPECT_TYPES:
                                t_rcv = calc_aspect_time(recv_pos, other_pos, a2, chart.julian_day, days_ahead)
                                if _valid(t_rcv, recv_pos, other_pos) and t_rcv < t_event:
                                    _record_event(
                                        {
                                            "t": t_rcv,
                                            "payload": {
                                                "prohibited": True,
                                                "type": "abscission",
                                                "abscissor": other,
                                                "significator": receiver_planet,
                                                "t_abscission": t_rcv,
                                                "reason": f"{other.value} cuts off light to {receiver_planet.value} before collection",
                                            },
                                        }
                                    )

                        _record_event(
                            {
                                "t": t_event,
                                "payload": {
                                    "prohibited": False,
                                    "type": "collection",
                                    "collector": planet,
                                    "t_event": t_event,
                                    "aspect": aspect,
                                    "quality": quality,
                                    "reception": has_reception,
                                    "reason": reason,
                                },
                            }
                        )
                    else:
                        if t1 > 0 and (t1 <= t2 or t2 <= 0):
                            _record_event(
                                {
                                    "t": t1,
                                    "payload": {
                                        "prohibited": True,
                                        "type": "prohibition",
                                        "prohibitor": planet,
                                        "significator": sig1,
                                        "t_prohibition": t1,
                                        "reason": f"{planet.value} {verb(aspect)} {sig1.value} before perfection",
                                    },
                                }
                            )
                        elif t2 > 0 and (t2 < t1 or t1 <= 0):
                            _record_event(
                                {
                                    "t": t2,
                                    "payload": {
                                        "prohibited": True,
                                        "type": "prohibition",
                                        "prohibitor": planet,
                                        "significator": sig2,
                                        "t_prohibition": t2,
                                        "reason": f"{planet.value} {verb(aspect)} {sig2.value} before perfection",
                                    },
                                }
                            )
            elif valid1 and t1 > 0:
                _record_event(
                    {
                        "t": t1,
                        "payload": {
                            "prohibited": True,
                            "type": "prohibition",
                            "prohibitor": planet,
                            "significator": sig1,
                            "t_prohibition": t1,
                            "reason": f"{planet.value} {verb(aspect)} {sig1.value} before perfection",
                        },
                    }
                )
            elif valid2 and t2 > 0:
                print(f"*** PROHIBITION DETECTED: {planet.value} {aspect.value} {sig2.value} in {t2} days ***")
                _record_event(
                    {
                        "t": t2,
                        "payload": {
                            "prohibited": True,
                            "type": "prohibition",
                            "prohibitor": planet,
                            "significator": sig2,
                            "t_prohibition": t2,
                            "reason": f"{planet.value} {verb(aspect)} {sig2.value} before perfection",
                        },
                    }
                )

    chosen_event: Optional[Dict[str, Any]] = None
    if earliest_prohibition and (
        earliest_tc_event is None or earliest_prohibition["t"] < earliest_tc_event["t"]
    ):
        chosen_event = earliest_prohibition
    elif earliest_tc_event:
        chosen_event = earliest_tc_event

    if chosen_event:
        print(f"Final chosen event: {chosen_event['payload']}")
        return chosen_event["payload"]

    print("Final chosen event: none (no prohibitions detected)")
    return {"prohibited": False, "type": "none", "reason": "No prohibitions detected"}
