"""Serialization helpers for the horary engine."""

from typing import Any, Dict, Optional
import datetime
import math

import swisseph as swe

try:
    from ..models import (
        HoraryChart,
        LunarAspect,
        Planet,
        PlanetPosition,
        SolarAnalysis,
        SolarCondition,
        Aspect as AspectType,
        Sign,
        AspectInfo,
    )
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import (
        HoraryChart,
        LunarAspect,
        Planet,
        PlanetPosition,
        SolarAnalysis,
        SolarCondition,
        Aspect as AspectType,
        Sign,
        AspectInfo,
    )

try:
    from .dsl import (
        Abscission,
        AccidentalDignity,
        Aspect as DslAspect,
        Collection,
        EssentialDignity,
        Frustration,
        HousePlacement,
        MoonVoidOfCourse,
        Prohibition,
        Reception,
        Refranation,
        Role,
        RoleImportance,
        Translation,
    )
except ImportError:  # pragma: no cover
    from dsl import (
        Abscission,
        AccidentalDignity,
        Aspect as DslAspect,
        Collection,
        EssentialDignity,
        Frustration,
        HousePlacement,
        MoonVoidOfCourse,
        Prohibition,
        Reception,
        Refranation,
        Role,
        RoleImportance,
        Translation,
    )


def serialize_lunar_aspect(lunar_aspect: Optional[LunarAspect]) -> Optional[Dict]:
    """Serialize a LunarAspect into a dictionary"""
    if not lunar_aspect:
        return None
    return {
        "planet": lunar_aspect.planet.value,
        "aspect": lunar_aspect.aspect.display_name,
        "orb": round(lunar_aspect.orb, 2),
        "degrees_difference": round(lunar_aspect.degrees_difference, 2),
        "perfection_eta_days": round(lunar_aspect.perfection_eta_days, 2),
        "perfection_eta_description": lunar_aspect.perfection_eta_description,
        "applying": lunar_aspect.applying,
    }


def serialize_planet_with_solar(
    planet_pos: PlanetPosition, solar_analysis: Optional[SolarAnalysis] = None
) -> Dict:
    """Enhanced helper function to serialize planet data including solar conditions"""
    data = {
        "longitude": float(planet_pos.longitude),
        "latitude": float(planet_pos.latitude),
        "house": int(planet_pos.house),
        "sign": planet_pos.sign.sign_name,
        "dignity_score": int(planet_pos.dignity_score),
        "essential_dignity": int(planet_pos.essential_dignity),
        "accidental_dignity": int(planet_pos.accidental_dignity),
        "retrograde": bool(planet_pos.retrograde),
        "speed": float(planet_pos.speed),
        "degree_in_sign": float(planet_pos.longitude % 30),
    }

    if solar_analysis:
        data["solar_condition"] = {
            "condition": solar_analysis.condition.condition_name,
            "distance_from_sun": round(solar_analysis.distance_from_sun, 4),
            "dignity_effect": solar_analysis.condition.dignity_modifier,
            "description": solar_analysis.condition.description,
            "exact_cazimi": solar_analysis.exact_cazimi,
            "traditional_exception": solar_analysis.traditional_exception,
        }

    return data


def serialize_chart_for_frontend(
    chart: HoraryChart, solar_analyses: Optional[Dict[Planet, SolarAnalysis]] = None
) -> Dict[str, Any]:
    """Enhanced serialize HoraryChart object for frontend consumption"""

    planets_data: Dict[str, Any] = {}
    for planet, planet_pos in chart.planets.items():
        solar_analysis = solar_analyses.get(planet) if solar_analyses else None
        planets_data[planet.value] = serialize_planet_with_solar(planet_pos, solar_analysis)

    aspects_data = []
    for aspect in chart.aspects:
        aspects_data.append(
            {
                "planet1": aspect.planet1.value,
                "planet2": aspect.planet2.value,
                "aspect": aspect.aspect.display_name,
                "orb": round(aspect.orb, 2),
                "applying": aspect.applying,
                "time_to_perfection": round(aspect.time_to_perfection, 2)
                if math.isfinite(aspect.time_to_perfection)
                else None,
                "perfection_within_sign": aspect.perfection_within_sign,
                "degrees_to_exact": round(aspect.degrees_to_exact, 2),
                "exact_time": aspect.exact_time.isoformat() if aspect.exact_time else None,
            }
        )

    solar_conditions_summary = None
    if solar_analyses:
        cazimi_planets = []
        combusted_planets = []
        under_beams_planets = []
        free_planets = []

        for planet, analysis in solar_analyses.items():
            planet_info = {
                "planet": planet.value,
                "distance_from_sun": round(analysis.distance_from_sun, 4),
            }

            if analysis.condition == SolarCondition.CAZIMI:
                planet_info["exact_cazimi"] = analysis.exact_cazimi
                planet_info["dignity_effect"] = analysis.condition.dignity_modifier
                cazimi_planets.append(planet_info)
            elif analysis.condition == SolarCondition.COMBUSTION:
                planet_info["traditional_exception"] = analysis.traditional_exception
                planet_info["dignity_effect"] = analysis.condition.dignity_modifier
                combusted_planets.append(planet_info)
            elif analysis.condition == SolarCondition.UNDER_BEAMS:
                planet_info["dignity_effect"] = analysis.condition.dignity_modifier
                under_beams_planets.append(planet_info)
            else:  # FREE
                free_planets.append(planet_info)

        solar_conditions_summary = {
            "cazimi_planets": cazimi_planets,
            "combusted_planets": combusted_planets,
            "under_beams_planets": under_beams_planets,
            "free_planets": free_planets,
            "significant_conditions": len(cazimi_planets)
            + len(combusted_planets)
            + len(under_beams_planets),
        }

    result: Dict[str, Any] = {
        "planets": planets_data,
        "aspects": aspects_data,
        "houses": [round(cusp, 2) for cusp in chart.houses],
        "house_rulers": {str(house): ruler.value for house, ruler in chart.house_rulers.items()},
        "ascendant": round(chart.ascendant, 4),
        "midheaven": round(chart.midheaven, 4),
        "solar_conditions_summary": solar_conditions_summary,
        "timezone_info": {
            "local_time": chart.date_time.isoformat(),
            "utc_time": chart.date_time_utc.isoformat(),
            "timezone": chart.timezone_info,
            "location_name": chart.location_name,
            "coordinates": {
                "latitude": chart.location[0],
                "longitude": chart.location[1],
            },
        },
    }

    if hasattr(chart, "moon_last_aspect") and chart.moon_last_aspect:
        result["moon_last_aspect"] = serialize_lunar_aspect(chart.moon_last_aspect)

    if hasattr(chart, "moon_next_aspect") and chart.moon_next_aspect:
        result["moon_next_aspect"] = serialize_lunar_aspect(chart.moon_next_aspect)

    return result


def deserialize_chart_for_evaluation(data: Dict[str, Any]) -> HoraryChart:
    """Deserialize serialized chart data into a :class:`HoraryChart`.

    The input ``data`` is expected to follow the structure produced by
    :func:`serialize_chart_for_frontend`. Only the fields required by the
    evaluation pipeline are reconstructed.
    """

    tz_info = data.get("timezone_info", {})
    dt_local = datetime.datetime.fromisoformat(tz_info.get("local_time"))
    dt_utc = datetime.datetime.fromisoformat(tz_info.get("utc_time"))
    lat = tz_info.get("coordinates", {}).get("latitude", 0.0)
    lon = tz_info.get("coordinates", {}).get("longitude", 0.0)

    planets: Dict[Planet, PlanetPosition] = {}
    solar_analyses: Dict[Planet, SolarAnalysis] = {}
    for name, p in data.get("planets", {}).items():
        planet_enum = Planet[name.upper()]
        sign_enum = Sign[p["sign"].upper()]
        planets[planet_enum] = PlanetPosition(
            planet=planet_enum,
            longitude=p["longitude"],
            latitude=p["latitude"],
            house=p["house"],
            sign=sign_enum,
            dignity_score=p.get("dignity_score", 0),
            essential_dignity=p.get("essential_dignity", 0),
            accidental_dignity=p.get("accidental_dignity", 0),
            retrograde=p.get("retrograde", False),
            speed=p.get("speed", 0.0),
            dignities=p.get("dignities", []),
        )
        sc = p.get("solar_condition")
        if sc:
            cond_name = sc["condition"].replace(" ", "_").upper()
            if cond_name == "FREE_OF_SUN":
                cond_name = "FREE"
            elif cond_name == "UNDER_THE_BEAMS":
                cond_name = "UNDER_BEAMS"
            solar_analyses[planet_enum] = SolarAnalysis(
                planet=planet_enum,
                distance_from_sun=sc["distance_from_sun"],
                condition=SolarCondition[cond_name],
                exact_cazimi=sc.get("exact_cazimi", False),
                traditional_exception=sc.get("traditional_exception", False),
            )

    aspects = []
    for a in data.get("aspects", []):
        aspects.append(
            AspectInfo(
                planet1=Planet[a["planet1"].upper()],
                planet2=Planet[a["planet2"].upper()],
                aspect=AspectType[a["aspect"].upper()],
                orb=a["orb"],
                applying=a.get("applying", False),
                time_to_perfection=a.get("time_to_perfection", math.inf),
                perfection_within_sign=a.get("perfection_within_sign", True),
                exact_time=datetime.datetime.fromisoformat(a["exact_time"])
                if a.get("exact_time")
                else None,
                degrees_to_exact=a.get("degrees_to_exact", 0.0),
            )
        )

    house_rulers = {
        int(h): Planet[r.upper()] for h, r in data.get("house_rulers", {}).items()
    }

    moon_last_aspect = None
    if ml := data.get("moon_last_aspect"):
        moon_last_aspect = LunarAspect(
            planet=Planet[ml["planet"].upper()],
            aspect=AspectType[ml["aspect"].upper()],
            orb=ml["orb"],
            degrees_difference=ml["degrees_difference"],
            perfection_eta_days=ml["perfection_eta_days"],
            perfection_eta_description=ml["perfection_eta_description"],
            applying=ml["applying"],
        )

    moon_next_aspect = None
    if mn := data.get("moon_next_aspect"):
        moon_next_aspect = LunarAspect(
            planet=Planet[mn["planet"].upper()],
            aspect=AspectType[mn["aspect"].upper()],
            orb=mn["orb"],
            degrees_difference=mn["degrees_difference"],
            perfection_eta_days=mn["perfection_eta_days"],
            perfection_eta_description=mn["perfection_eta_description"],
            applying=mn["applying"],
        )

    jd = swe.julday(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0,
    )

    return HoraryChart(
        date_time=dt_local,
        date_time_utc=dt_utc,
        timezone_info=tz_info.get("timezone", "UTC"),
        location=(lat, lon),
        location_name=tz_info.get("location_name", ""),
        planets=planets,
        aspects=aspects,
        houses=data.get("houses", []),
        house_rulers=house_rulers,
        ascendant=data.get("ascendant", 0.0),
        midheaven=data.get("midheaven", 0.0),
        solar_analyses=solar_analyses or None,
        julian_day=jd,
        moon_last_aspect=moon_last_aspect,
        moon_next_aspect=moon_next_aspect,
    )


# ---------------------------------------------------------------------------
# DSL primitive serialization
# ---------------------------------------------------------------------------


def _actor_to_json(actor: Role | Planet) -> Dict[str, str]:
    """Serialize an actor (Planet or Role) to a minimal JSON form."""
    if isinstance(actor, Planet):
        return {"planet": actor.value}
    return {"role": actor.name}


def _actor_from_json(data: Dict[str, str]) -> Role | Planet:
    """Deserialize an actor from its JSON representation."""
    if "planet" in data:
        return Planet(data["planet"])
    return Role(data["role"])


def serialize_primitive(primitive: Any) -> Dict[str, Any]:
    """Serialize a DSL primitive into a concise dictionary."""
    if isinstance(primitive, DslAspect):
        return {
            "type": "aspect",
            "a1": _actor_to_json(primitive.actor1),
            "a2": _actor_to_json(primitive.actor2),
            "aspect": primitive.aspect.name,
            "app": primitive.applying,
        }
    if isinstance(primitive, Translation):
        return {
            "type": "translation",
            "tr": _actor_to_json(primitive.translator),
            "frm": _actor_to_json(primitive.from_actor),
            "to": _actor_to_json(primitive.to_actor),
            "asp": primitive.aspect.name,
            "rcpt": primitive.reception,
            "app": primitive.applying,
        }
    if isinstance(primitive, Collection):
        return {
            "type": "collection",
            "col": _actor_to_json(primitive.collector),
            "a1": _actor_to_json(primitive.actor1),
            "a2": _actor_to_json(primitive.actor2),
            "asp": primitive.aspect.name,
            "rcpt": primitive.reception,
            "app": primitive.applying,
        }
    if isinstance(primitive, Prohibition):
        return {
            "type": "prohibition",
            "pr": _actor_to_json(primitive.prohibitor),
            "sig": _actor_to_json(primitive.significator),
            "asp": primitive.aspect.name if primitive.aspect else None,
        }
    if isinstance(primitive, Refranation):
        return {
            "type": "refranation",
            "rf": _actor_to_json(primitive.refrainer),
            "oth": _actor_to_json(primitive.other),
        }
    if isinstance(primitive, Frustration):
        return {
            "type": "frustration",
            "fr": _actor_to_json(primitive.frustrator),
            "frm": _actor_to_json(primitive.from_actor),
            "to": _actor_to_json(primitive.to_actor),
        }
    if isinstance(primitive, Abscission):
        return {
            "type": "abscission",
            "ab": _actor_to_json(primitive.abscissor),
            "frm": _actor_to_json(primitive.from_actor),
            "to": _actor_to_json(primitive.to_actor),
        }
    if isinstance(primitive, Reception):
        return {
            "type": "reception",
            "rcv": _actor_to_json(primitive.receiver),
            "rcd": _actor_to_json(primitive.received),
            "dig": primitive.dignity,
        }
    if isinstance(primitive, EssentialDignity):
        return {
            "type": "essential",
            "act": _actor_to_json(primitive.actor),
            "score": primitive.score,
        }
    if isinstance(primitive, AccidentalDignity):
        return {
            "type": "accidental",
            "act": _actor_to_json(primitive.actor),
            "score": primitive.score,
        }
    if isinstance(primitive, MoonVoidOfCourse):
        return {
            "type": "moon_voc",
            "voc": primitive.is_voc,
            "detail": primitive.detail,
        }
    if isinstance(primitive, HousePlacement):
        return {
            "type": "house",
            "act": _actor_to_json(primitive.actor),
            "house": primitive.house,
        }
    if isinstance(primitive, RoleImportance):
        return {
            "type": "role_importance",
            "role": primitive.role.name,
            "importance": primitive.importance,
        }
    raise TypeError(f"Unsupported primitive type: {type(primitive)!r}")


def deserialize_primitive(data: Dict[str, Any]) -> Any:
    """Reconstruct a DSL primitive from its serialized form."""
    t = data["type"]
    if t == "aspect":
        return DslAspect(
            _actor_from_json(data["a1"]),
            _actor_from_json(data["a2"]),
            AspectType[data["aspect"]],
            data.get("app", True),
        )
    if t == "translation":
        return Translation(
            _actor_from_json(data["tr"]),
            _actor_from_json(data["frm"]),
            _actor_from_json(data["to"]),
            data.get("app", True),
            AspectType[data.get("asp", "CONJUNCTION")],
            data.get("rcpt", False),
        )
    if t == "collection":
        return Collection(
            _actor_from_json(data["col"]),
            _actor_from_json(data["a1"]),
            _actor_from_json(data["a2"]),
            data.get("app", True),
            AspectType[data.get("asp", "CONJUNCTION")],
            data.get("rcpt", False),
        )
    if t == "prohibition":
        asp = data.get("asp")
        aspect_enum = AspectType[asp] if asp is not None else None
        return Prohibition(
            _actor_from_json(data["pr"]),
            _actor_from_json(data["sig"]),
            aspect_enum,
        )
    if t == "refranation":
        return Refranation(
            _actor_from_json(data["rf"]),
            _actor_from_json(data["oth"]),
        )
    if t == "frustration":
        return Frustration(
            _actor_from_json(data["fr"]),
            _actor_from_json(data["frm"]),
            _actor_from_json(data["to"]),
        )
    if t == "abscission":
        return Abscission(
            _actor_from_json(data["ab"]),
            _actor_from_json(data["frm"]),
            _actor_from_json(data["to"]),
        )
    if t == "reception":
        return Reception(
            _actor_from_json(data["rcv"]),
            _actor_from_json(data["rcd"]),
            data["dig"],
        )
    if t == "essential":
        return EssentialDignity(
            _actor_from_json(data["act"]),
            data["score"],
        )
    if t == "accidental":
        return AccidentalDignity(
            _actor_from_json(data["act"]),
            data["score"],
        )
    if t == "moon_voc":
        return MoonVoidOfCourse(
            bool(data["voc"]),
            data.get("detail"),
        )
    if t == "house":
        return HousePlacement(
            _actor_from_json(data["act"]),
            int(data["house"]),
        )
    if t == "role_importance":
        return RoleImportance(Role(data["role"]), float(data["importance"]))
    raise ValueError(f"Unknown primitive type: {t}")
