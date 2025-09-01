import logging
from typing import Optional, Tuple

import datetime
import pytz
try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9
    ZoneInfo = None

try:
    from timezonefinder import TimezoneFinder
    TIMEZONEFINDER_AVAILABLE = True
except ImportError:  # pragma: no cover
    TimezoneFinder = None
    TIMEZONEFINDER_AVAILABLE = False

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable


logger = logging.getLogger(__name__)


class LocationError(Exception):
    """Custom exception for geocoding failures."""
    pass


def safe_geocode(location_string: str, timeout: int = 10) -> Tuple[float, float, str]:
    """Geocode a location string with fail-fast behaviour.

    Args:
        location_string: Location to geocode.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (latitude, longitude, full_address).

    Raises:
        LocationError: If geocoding fails or the library is unavailable.
    """
    try:
        geolocator = Nominatim(user_agent="horary_astrology_precise")
        location = geolocator.geocode(location_string, timeout=timeout)
        if location is None:
            raise LocationError(
                f"Location not found: '{location_string}'. Please provide a more specific location."
            )
        return (location.latitude, location.longitude, location.address)
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        raise LocationError(f"Geocoding service unavailable: {e}")
    except ImportError:
        raise LocationError("Geocoding library not available. Please install geopy.")
    except Exception as e:  # pragma: no cover - unexpected errors
        raise LocationError(f"Geocoding failed for '{location_string}': {e}")


class TimezoneManager:
    """Handles timezone operations for horary calculations."""

    def __init__(self) -> None:
        if TIMEZONEFINDER_AVAILABLE:
            try:
                self.tf = TimezoneFinder()
                logger.info("TimezoneFinder initialized successfully")
            except Exception as e:  # pragma: no cover - initialization failure
                logger.error(f"Failed to initialize TimezoneFinder: {e}")
                self.tf = None
        else:  # pragma: no cover - library missing
            logger.warning(
                "TimezoneFinder library not available - using fallback timezone detection only"
            )
            self.tf = None

        try:
            self.geolocator = Nominatim(user_agent="horary_astrology_tz")
            logger.info("Geolocator initialized successfully")
        except Exception as e:  # pragma: no cover - geolocator failure
            logger.error(f"Failed to initialize Geolocator: {e}")
            self.geolocator = None

    def get_timezone_for_location(self, lat: float, lon: float) -> Optional[str]:
        """Get timezone string for given coordinates with enhanced debugging."""
        logger.info(f"=== TIMEZONE DETECTION STARTED for {lat}, {lon} ===")

        try:
            if self.tf is not None:
                logger.info("Using TimezoneFinder library")
                timezone_result = self.tf.timezone_at(lat=lat, lng=lon)
                logger.info(f"TimezoneFinder raw result: {timezone_result}")

                if timezone_result:
                    logger.info("Validating timezone result...")
                    validated_tz = self._validate_timezone_for_coordinates(
                        timezone_result, lat, lon
                    )
                    if validated_tz:
                        logger.info(
                            f"=== FINAL TIMEZONE: {validated_tz} (after validation) ==="
                        )
                        return validated_tz
            else:
                logger.info(
                    f"TimezoneFinder not available, using fallback for {lat}, {lon}"
                )
                timezone_result = None

            fallback_tz = self._get_fallback_timezone(lat, lon)
            if fallback_tz:
                logger.warning(
                    f"Using fallback timezone {fallback_tz} for {lat}, {lon} (TimezoneFinder returned: {timezone_result})"
                )
                return fallback_tz

            return timezone_result
        except Exception as e:  # pragma: no cover - unexpected errors
            logger.error(f"Error getting timezone for {lat}, {lon}: {e}")
            try:
                fallback_tz = self._get_fallback_timezone(lat, lon)
                if fallback_tz:
                    logger.warning(
                        f"Using fallback timezone {fallback_tz} after TimezoneFinder error"
                    )
                    return fallback_tz
            except Exception:
                pass
            return None

    def _validate_timezone_for_coordinates(
        self, timezone_str: str, lat: float, lon: float
    ) -> Optional[str]:
        """Validate that timezone makes geographic sense for coordinates."""

        logger.info(
            f"TIMEZONE VALIDATION: Checking {timezone_str} for coordinates {lat}, {lon}"
        )

        geographic_validations = {
            (29.5, 33.5, 34.0, 36.0): "Asia/Jerusalem",
        }

        for (lat_min, lat_max, lon_min, lon_max), expected_tz in geographic_validations.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                logger.info(
                    f"COORDINATE MATCH: {lat},{lon} falls in range {lat_min}-{lat_max}, {lon_min}-{lon_max}"
                )
                if timezone_str != expected_tz:
                    logger.warning(
                        f"TIMEZONE OVERRIDE: TimezoneFinder returned {timezone_str} for {lat},{lon} but expected {expected_tz} - CORRECTING"
                    )
                    return expected_tz
                else:
                    logger.info(
                        f"TIMEZONE OK: {timezone_str} is correct for coordinates {lat},{lon}"
                    )
                    return timezone_str

        suspicious_combinations = [
            (29.5, 33.5, 34.0, 36.0, ["America/"])
        ]

        for (
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            forbidden_prefixes,
        ) in suspicious_combinations:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                if any(timezone_str.startswith(prefix) for prefix in forbidden_prefixes):
                    logger.warning(
                        f"TIMEZONE OVERRIDE: Suspicious combination {lat},{lon} with timezone {timezone_str}"
                    )
                    return geographic_validations.get(
                        (29.5, 33.5, 34.0, 36.0), timezone_str
                    )

        return timezone_str

    def _get_fallback_timezone(self, lat: float, lon: float) -> Optional[str]:
        """Fallback method to get timezone if TimezoneFinder fails."""
        if self.geolocator is None:
            return None

        try:
            location = self.geolocator.reverse((lat, lon), exactly_one=True)
            if location and location.raw.get("address"):
                address = location.raw["address"]
                if "country_code" in address:
                    country_code = address["country_code"].upper()
                    country = pytz.country_names.get(country_code)
                    if country:
                        timezones = pytz.country_timezones.get(country_code)
                        if timezones:
                            logger.info(
                                f"Using country-level timezone fallback {timezones[0]} for {lat}, {lon}"
                            )
                            return timezones[0]
        except (GeocoderTimedOut, GeocoderUnavailable):
            logger.warning("Geolocator reverse lookup timed out")
        except Exception as e:  # pragma: no cover - unexpected errors
            logger.error(f"Fallback timezone detection failed: {e}")

        try:
            from timezonefinder import TimezoneFinderL

            tf = TimezoneFinder()
            return tf.timezone_at(lat=lat, lng=lon)
        except Exception:
            return None

    def parse_datetime_with_timezone(
        self,
        date_str: str,
        time_str: str,
        timezone_str: Optional[str] = None,
        lat: float = None,
        lon: float = None,
    ) -> Tuple[datetime.datetime, datetime.datetime, str]:
        """Parse datetime string and return both local and UTC datetime objects."""
        datetime_str = f"{date_str} {time_str}"

        date_formats = [
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M",
            "%m/%d/%Y %H:%M",
            "%d-%m-%Y %H:%M",
            "%Y/%m/%d %H:%M",
        ]

        dt_naive = None
        format_used = None
        for date_format in date_formats:
            try:
                dt_naive = datetime.datetime.strptime(datetime_str, date_format)
                format_used = date_format
                logger.info(
                    f"Successfully parsed datetime '{datetime_str}' using format '{date_format}'"
                )
                break
            except ValueError:
                continue

        if dt_naive is None:
            raise ValueError(
                f"Unable to parse date '{date_str}'. Please use DD/MM/YYYY format (e.g., 02/03/2004 for March 2, 2004)"
            )

        logger.info(f"Parsed date: {dt_naive} using format: {format_used}")

        if timezone_str:
            try:
                if ZoneInfo:
                    tz = ZoneInfo(timezone_str)
                else:
                    tz = pytz.timezone(timezone_str)
                timezone_used = timezone_str
            except Exception:
                tz = pytz.UTC
                timezone_used = "UTC"
        elif lat is not None and lon is not None:
            tz_str = self.get_timezone_for_location(lat, lon)
            if tz_str:
                try:
                    if ZoneInfo:
                        tz = ZoneInfo(tz_str)
                    else:
                        tz = pytz.timezone(tz_str)
                    timezone_used = tz_str
                except Exception:
                    tz = pytz.UTC
                    timezone_used = "UTC"
            else:
                tz = pytz.UTC
                timezone_used = "UTC"
        else:
            tz = pytz.UTC
            timezone_used = "UTC"

        if ZoneInfo or hasattr(tz, "localize"):
            if hasattr(tz, "localize"):
                try:
                    dt_local = tz.localize(dt_naive)
                except pytz.AmbiguousTimeError:
                    dt_local = tz.localize(dt_naive, is_dst=False)
                    logger.warning(
                        f"Ambiguous time {dt_naive} - using standard time"
                    )
                except pytz.NonExistentTimeError:
                    dt_adjusted = dt_naive + datetime.timedelta(hours=1)
                    dt_local = tz.localize(dt_adjusted)
                    logger.warning(
                        f"Non-existent time {dt_naive} - using {dt_adjusted}"
                    )
            else:
                dt_local = dt_naive.replace(tzinfo=tz)
        else:
            dt_local = dt_naive.replace(tzinfo=tz)

        dt_utc = dt_local.astimezone(pytz.UTC)
        return dt_local, dt_utc, timezone_used

    def get_current_time_for_location(
        self, lat: float, lon: float
    ) -> Tuple[datetime.datetime, datetime.datetime, str]:
        """Get current time for a specific location."""
        tz_str = self.get_timezone_for_location(lat, lon)

        if tz_str:
            try:
                if ZoneInfo:
                    tz = ZoneInfo(tz_str)
                else:
                    tz = pytz.timezone(tz_str)
                timezone_used = tz_str
            except Exception:
                tz = pytz.UTC
                timezone_used = "UTC"
        else:
            tz = pytz.UTC
            timezone_used = "UTC"

        utc_now = datetime.datetime.now(pytz.UTC)
        local_now = utc_now.astimezone(tz)

        return local_now, utc_now, timezone_used
