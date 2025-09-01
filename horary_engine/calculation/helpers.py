# -*- coding: utf-8 -*-
"""
Created on Sat May 31 13:30:09 2025

@author: sabaa
"""

# -*- coding: utf-8 -*-
"""
Traditional Horary Astrology Mathematical Helpers
Created for computational functions used in horary judgment

@author: horary_engine_extension
"""

import math
import datetime
from typing import Tuple, Optional, Dict, Any
import swisseph as swe


def calculate_next_station_time(planet_id: int, jd_start: float, 
                               max_days: int = 365) -> Optional[float]:
    """
    Calculate when a planet will next station (turn retrograde/direct)
    using Swiss Ephemeris.
    
    Args:
        planet_id: Swiss Ephemeris planet ID
        jd_start: Starting Julian Day 
        max_days: Maximum days to search ahead
    
    Returns:
        Julian Day of next station, or None if not found
    
    Classical source: Lilly III Chap. XXI - "Of the frustration of Planets"
    """
    step_size = 0.1  # Check every 0.1 days
    
    try:
        # Get initial speed
        initial_data, _ = swe.calc_ut(jd_start, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
        initial_speed = initial_data[3]
        
        # Search forward in time
        current_jd = jd_start + step_size
        max_jd = jd_start + max_days
        
        previous_speed = initial_speed
        
        while current_jd < max_jd:
            try:
                planet_data, _ = swe.calc_ut(current_jd, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
                current_speed = planet_data[3]
                
                # Check for sign change in speed (station)
                if (previous_speed > 0 and current_speed < 0) or (previous_speed < 0 and current_speed > 0):
                    # Found a station, refine the timing
                    return _refine_station_time(planet_id, current_jd - step_size, current_jd)
                
                previous_speed = current_speed
                current_jd += step_size
                
            except Exception:
                # Skip errors and continue
                current_jd += step_size
                continue
    
    except Exception:
        pass
    
    return None


def _refine_station_time(planet_id: int, jd_before: float, jd_after: float) -> float:
    """Refine station time to higher precision using binary search"""
    tolerance = 0.001  # About 1.5 minutes
    
    while (jd_after - jd_before) > tolerance:
        jd_mid = (jd_before + jd_after) / 2
        
        try:
            data_before, _ = swe.calc_ut(jd_before, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
            data_mid, _ = swe.calc_ut(jd_mid, planet_id, swe.FLG_SWIEPH | swe.FLG_SPEED)
            
            speed_before = data_before[3]
            speed_mid = data_mid[3]
            
            # Check which side of midpoint the station is on
            if (speed_before > 0 and speed_mid > 0) or (speed_before < 0 and speed_mid < 0):
                # Station is after midpoint
                jd_before = jd_mid
            else:
                # Station is before midpoint
                jd_after = jd_mid
                
        except Exception:
            break
    
    return (jd_before + jd_after) / 2


def calculate_future_longitude(longitude: float, speed: float, days: float) -> float:
    """
    Calculate where a planet will be in the future given current position and speed.

    Args:
        longitude: Current longitude in degrees
        speed: Current speed in degrees per day (negative for retrograde)
        days: Number of days in the future

    Returns:
        Future longitude in degrees (0-360)

    Classical source: Ptolemy Tetrabiblos - planetary motion calculations
    """
    future_longitude = longitude + (speed * days)

    # Normalize to 0-360 degrees
    return future_longitude % 360


def calculate_sign_boundary_longitude(current_longitude: float, direction: int) -> float:
    """
    Calculate the longitude of the next sign boundary in the direction of motion.
    
    Args:
        current_longitude: Current longitude in degrees
        direction: +1 for direct motion, -1 for retrograde motion
    
    Returns:
        Longitude of next sign boundary in direction of motion
    
    Classical source: Firmicus Maternus - sign boundaries and planetary motion
    """
    current_longitude = current_longitude % 360
    current_sign_start = (int(current_longitude // 30)) * 30
    
    if direction > 0:  # Direct motion - next sign forward
        next_boundary = current_sign_start + 30
        if next_boundary >= 360:
            next_boundary = 0
    else:  # Retrograde motion - previous sign backward
        next_boundary = current_sign_start
        if abs(current_longitude - current_sign_start) < 1e-6:  # Exactly on boundary
            next_boundary = current_sign_start - 30
            if next_boundary < 0:
                next_boundary = 330
    
    return next_boundary


def days_to_sign_exit(longitude: float, speed: float) -> Optional[float]:
    """
    Calculate days until planet exits current sign based on motion direction.
    
    Args:
        longitude: Current longitude in degrees
        speed: Speed in degrees per day (negative for retrograde)
    
    Returns:
        Days until sign exit, or None if stationary
    
    Classical source: Lilly III Chap. XXV - "Of timing in horary questions"
    """
    if abs(speed) < 0.001:  # Nearly stationary
        return None
    
    direction = 1 if speed > 0 else -1
    boundary_longitude = calculate_sign_boundary_longitude(longitude, direction)
    
    # Calculate degrees to boundary
    if direction > 0:  # Direct motion
        if boundary_longitude > longitude:
            degrees_to_boundary = boundary_longitude - longitude
        else:  # Crossing 0° Aries
            degrees_to_boundary = (360 - longitude) + boundary_longitude
    else:  # Retrograde motion
        if boundary_longitude < longitude:
            degrees_to_boundary = longitude - boundary_longitude
        else:  # Crossing from Aries to Pisces
            degrees_to_boundary = longitude + (360 - boundary_longitude)
    
    return degrees_to_boundary / abs(speed)


def is_within_sign_change(
    longitude: float, speed: float, threshold: float = 1.0
) -> bool:
    """Return True if a planet is close to a sign boundary and moving toward it.

    Parameters
    ----------
    longitude:
        Current ecliptic longitude of the planet in degrees.
    speed:
        Speed in degrees per day (negative for retrograde). A value near zero
        is treated as stationary and will always return ``False``.
    threshold:
        Number of degrees from the next sign boundary at which the planet
        should be flagged.

    Returns
    -------
    bool
        ``True`` if the planet is within ``threshold`` degrees of the next
        sign boundary *and* its motion carries it toward that boundary.
    """

    if abs(speed) < 0.001:
        return False

    direction = 1 if speed > 0 else -1
    boundary_longitude = calculate_sign_boundary_longitude(longitude, direction)

    if direction > 0:
        distance = (boundary_longitude - longitude) % 360
    else:
        distance = (longitude - boundary_longitude) % 360

    return distance <= threshold


def calculate_elongation(planet_longitude: float, sun_longitude: float) -> float:
    """
    Calculate elongation (angular distance) between planet and Sun.
    
    Args:
        planet_longitude: Planet's ecliptic longitude
        sun_longitude: Sun's ecliptic longitude
    
    Returns:
        Elongation in degrees (0-180)
    
    Classical source: Ptolemy Almagest - planetary visibility calculations
    """
    diff = abs(planet_longitude - sun_longitude)
    return min(diff, 360 - diff)


def is_planet_oriental(planet_longitude: float, sun_longitude: float) -> bool:
    """
    Determine if planet is oriental (rising before Sun) or occidental (setting after Sun).
    
    Args:
        planet_longitude: Planet's ecliptic longitude
        sun_longitude: Sun's ecliptic longitude
    
    Returns:
        True if oriental (morning star), False if occidental (evening star)
    
    Classical source: Ptolemy Tetrabiblos - oriental and occidental planets
    """
    # Normalize longitudes
    planet_lon = planet_longitude % 360
    sun_lon = sun_longitude % 360
    
    # Calculate relative position
    relative_position = (planet_lon - sun_lon) % 360
    
    # Oriental if planet is 0° to 180° ahead of Sun in zodiacal order
    return 0 < relative_position < 180


def sun_altitude_at_civil_twilight(latitude: float, longitude: float, 
                                  jd_ut: float) -> float:
    """
    Calculate Sun's altitude at civil twilight for visibility calculations.
    
    Args:
        latitude: Observer latitude in degrees
        longitude: Observer longitude in degrees  
        jd_ut: Julian Day (UT)
    
    Returns:
        Sun's altitude in degrees (negative below horizon)

    Classical source: Al-Biruni - planetary visibility and heliacal risings
    """
    try:
        # Calculate ecliptic position of the Sun
        sun_data, _ = swe.calc_ut(jd_ut, swe.SUN, swe.FLG_SWIEPH)
        sun_longitude = sun_data[0]
        sun_latitude = sun_data[1]
        sun_distance = sun_data[2]

        # Convert ecliptic coordinates to altitude/azimuth
        geopos = (longitude, latitude, 0)  # Observer position (east positive)
        _, altitude, _ = swe.azalt(
            jd_ut,
            swe.ECL2HOR,
            geopos,
            0,  # Atmospheric pressure (ignored)
            0,  # Atmospheric temperature (ignored)
            (sun_longitude, sun_latitude, sun_distance),
        )

        return float(altitude)

    except Exception:
        # Fallback to classical civil twilight threshold
        return -8.0


def calculate_moon_variable_speed(jd_ut: float) -> float:
    """
    Get Moon's current speed from ephemeris for variable timing calculations.
    
    Args:
        jd_ut: Julian Day (UT)
    
    Returns:
        Moon's speed in degrees per day
    
    Classical source: Lilly III Chap. XXV - Moon's variable motion in timing
    """
    try:
        moon_data, _ = swe.calc_ut(jd_ut, swe.MOON, swe.FLG_SWIEPH | swe.FLG_SPEED)
        return abs(moon_data[3])  # Return absolute speed
    except Exception:
        return 13.0  # Classical average fallback


def check_aspect_separation_order(
    planet_a_lon: float,
    planet_a_speed: float,
    planet_c_lon: float,
    planet_c_speed: float,
    aspect_degrees: float,
    jd_current: float,
) -> Dict[str, Any]:
    """Check if planet C is separating from an aspect with planet A.

    The previous implementation estimated the separation trend by looking one
    hour ahead. This fails for very slow or very fast planets. Instead we use
    the relative speed between the planets to analytically compute whether the
    orb is widening or narrowing, using the same approach as
    :func:`is_applying_enhanced`.

    Parameters
    ----------
    planet_a_lon : float
        Planet A longitude.
    planet_a_speed : float
        Planet A speed in degrees per day.
    planet_c_lon : float
        Planet C longitude.
    planet_c_speed : float
        Planet C speed in degrees per day.
    aspect_degrees : float
        Aspect angle (0, 60, 90, 120, 180).
    jd_current : float
        Current Julian Day (kept for API compatibility).

    Returns
    -------
    Dict[str, Any]
        Separation analysis containing the current orb and its rate of change.

    Classical source: Lilly III Chap. XXVI - Translation of Light
    """

    # Signed difference from exact aspect in range [-180, 180)
    diff = (planet_c_lon - planet_a_lon - aspect_degrees + 180) % 360 - 180

    # Current orb from exact aspect
    current_orb = abs(diff)

    # Relative speed determines whether the orb is widening or narrowing
    relative_speed = planet_c_speed - planet_a_speed
    orb_rate = diff * relative_speed
    is_separating = orb_rate > 0

    return {
        "is_separating": is_separating,
        "current_orb": current_orb,
        "orb_rate": orb_rate,
    }


def normalize_longitude(longitude: float) -> float:
    """Normalize longitude to 0-360 degrees"""
    return longitude % 360


def degrees_to_dms(degrees: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to degrees, minutes, seconds"""
    abs_deg = abs(degrees)
    deg = int(abs_deg)
    min_float = (abs_deg - deg) * 60
    min_int = int(min_float)
    sec = (min_float - min_int) * 60
    
    if degrees < 0:
        deg = -deg

    return (deg, min_int, sec)
