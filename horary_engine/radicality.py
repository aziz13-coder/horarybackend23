"""Radicality checks for horary charts."""

import datetime
from typing import Any, Dict

import swisseph as swe

from horary_config import cfg
try:
    from ..models import HoraryChart, Planet, Sign
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import HoraryChart, Planet, Sign


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
    0: Planet.MOON,  # Monday
    1: Planet.MARS,  # Tuesday
    2: Planet.MERCURY,  # Wednesday
    3: Planet.JUPITER,  # Thursday
    4: Planet.VENUS,  # Friday
    5: Planet.SATURN,  # Saturday
    6: Planet.SUN,  # Sunday
}


def _sign_triplicity(sign: Sign) -> str:
    """Return the elemental triplicity for a sign."""
    if sign in {Sign.ARIES, Sign.LEO, Sign.SAGITTARIUS}:
        return "fire"
    if sign in {Sign.TAURUS, Sign.VIRGO, Sign.CAPRICORN}:
        return "earth"
    if sign in {Sign.GEMINI, Sign.LIBRA, Sign.AQUARIUS}:
        return "air"
    return "water"


def check_planetary_hour_agreement(chart: HoraryChart, config) -> Dict[str, Any]:
    """Check if planetary hour ruler agrees with Ascendant ruler.

    Computes local sunrise and sunset using Swiss Ephemeris and divides the
    day/night into twelve unequal planetary hours. The active planetary hour at
    the query time determines the hour ruler.
    """

    # Ensure timezone-aware datetimes
    dt_local = chart.date_time
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=datetime.timezone.utc)
    dt_utc = chart.date_time_utc
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=datetime.timezone.utc)

    lat, lon = chart.location

    # Julian day numbers for computations
    jd = swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600,
        swe.GREG_CAL,
    )
    jd0 = swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, 0.0, swe.GREG_CAL)

    # Sunrise and sunset for today
    _, rise_today = swe.rise_trans(jd0, swe.SUN, swe.CALC_RISE, (lon, lat, 0))
    sunrise_today = rise_today[0]
    _, set_today = swe.rise_trans(jd0, swe.SUN, swe.CALC_SET, (lon, lat, 0))
    sunset_today = set_today[0]

    if jd >= sunrise_today:
        # After today's sunrise
        sunrise_prev = sunrise_today
        sunset_prev = sunset_today
        _, rise_next = swe.rise_trans(jd0 + 1, swe.SUN, swe.CALC_RISE, (lon, lat, 0))
        sunrise_next = rise_next[0]
    else:
        # Before today's sunrise -> use previous day's data
        _, rise_prev = swe.rise_trans(jd0 - 1, swe.SUN, swe.CALC_RISE, (lon, lat, 0))
        sunrise_prev = rise_prev[0]
        _, set_prev = swe.rise_trans(jd0 - 1, swe.SUN, swe.CALC_SET, (lon, lat, 0))
        sunset_prev = set_prev[0]
        sunrise_next = sunrise_today

    # Determine the planetary day ruler (day starts at local sunrise)
    y, m, d, h = swe.revjul(sunrise_today, swe.GREG_CAL)
    sunrise_today_dt = (
        datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc)
        + datetime.timedelta(hours=h)
    ).astimezone(dt_local.tzinfo)
    weekday = dt_local.weekday()
    if dt_local < sunrise_today_dt:
        weekday = (weekday - 1) % 7
    day_ruler = PLANETARY_DAY_RULERS.get(weekday, Planet.SUN)

    # Determine planetary hour index
    if sunrise_prev <= jd < sunset_prev:
        day_length = sunset_prev - sunrise_prev
        hour_len = day_length / 12.0
        hour_index = int((jd - sunrise_prev) / hour_len)
    else:
        night_length = sunrise_next - sunset_prev
        hour_len = night_length / 12.0
        hour_index = 12 + int((jd - sunset_prev) / hour_len)

    start_idx = PLANET_SEQUENCE.index(day_ruler)
    hour_ruler = PLANET_SEQUENCE[(start_idx + hour_index) % 7]

    asc_sign = list(Sign)[int((chart.ascendant % 360) // 30)]
    asc_ruler = asc_sign.ruler

    mode = getattr(config.radicality, "hour_agreement_mode", "ruler")

    if mode == "ruler":
        if hour_ruler == asc_ruler:
            return {
                "valid": True,
                "reason": f"Planetary hour ruler {hour_ruler.value} matches Ascendant ruler",
            }
        return {
            "valid": False,
            "reason": (
                f"Planetary hour ruler {hour_ruler.value} does not match Ascendant ruler {asc_ruler.value}"
            ),
        }

    if mode == "sign":
        hour_pos = chart.planets.get(hour_ruler)
        if hour_pos and hour_pos.sign == asc_sign:
            return {
                "valid": True,
                "reason": (
                    f"Planetary hour ruler {hour_ruler.value} in Ascendant sign {asc_sign.sign_name}"
                ),
            }
        return {
            "valid": False,
            "reason": (
                f"Planetary hour ruler {hour_ruler.value} not in Ascendant sign {asc_sign.sign_name}"
            ),
        }

    if mode == "triplicity":
        hour_pos = chart.planets.get(hour_ruler)
        if hour_pos and _sign_triplicity(hour_pos.sign) == _sign_triplicity(asc_sign):
            return {
                "valid": True,
                "reason": (
                    "Planetary hour ruler shares triplicity with Ascendant"
                ),
            }
        return {
            "valid": False,
            "reason": (
                "Planetary hour ruler does not share triplicity with Ascendant"
            ),
        }

    # Unsupported mode - default to valid
    return {"valid": True, "reason": "Unsupported hour agreement mode"}


def check_enhanced_radicality(chart: HoraryChart, ignore_saturn_7th: bool = False) -> Dict[str, Any]:
    """Enhanced radicality checks with configuration"""

    config = cfg()
    asc_degree = chart.ascendant % 30

    # Too early
    if asc_degree < config.radicality.asc_too_early:
        return {
            "valid": False,
            "reason": f"Ascendant too early at {asc_degree:.1f}° - question premature or not mature",
        }

    # Too late
    if asc_degree > config.radicality.asc_too_late:
        return {
            "valid": False,
            "reason": f"Ascendant too late at {asc_degree:.1f}° - question too late or already decided",
        }

    # Saturn in 7th house (configurable)
    if config.radicality.saturn_7th_enabled and not ignore_saturn_7th:
        saturn_pos = chart.planets[Planet.SATURN]
        if saturn_pos.house == 7:
            return {
                "valid": False,
                "reason": "Saturn in 7th house - astrologer may err in judgment (Bonatti)",
            }

    # Via Combusta (configurable)
    if config.radicality.via_combusta_enabled:
        moon_pos = chart.planets[Planet.MOON]
        moon_degree_in_sign = moon_pos.longitude % 30

        via_combusta = config.radicality.via_combusta

        if (
            (moon_pos.sign == Sign.LIBRA and moon_degree_in_sign > via_combusta.libra_start)
            or (
                moon_pos.sign == Sign.SCORPIO
                and moon_degree_in_sign <= via_combusta.scorpio_end
            )
        ):
            return {
                "valid": False,
                "reason": (
                    f"Moon in Via Combusta ({moon_pos.sign.sign_name} {moon_degree_in_sign:.1f}°) - "
                    "volatile or corrupted matter"
                ),
            }

    # Planetary hour agreement (configurable)
    if getattr(config.radicality, "hour_agreement_enabled", False):
        hour_check = check_planetary_hour_agreement(chart, config)
        if not hour_check["valid"]:
            return hour_check

    return {
        "valid": True,
        "reason": f"Chart is radical - Ascendant at {asc_degree:.1f}°",
    }
