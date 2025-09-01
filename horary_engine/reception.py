"""Reception calculations for the horary engine."""

from typing import Dict, List, Tuple, Any

from horary_config import cfg

try:
    from ..models import Planet, Sign, HoraryChart
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import Planet, Sign, HoraryChart


class TraditionalReceptionCalculator:
    """Centralized reception calculator - single source of truth for all reception logic"""

    def __init__(self) -> None:
        # Traditional exaltations
        self.exaltations = {
            Planet.SUN: Sign.ARIES,
            Planet.MOON: Sign.TAURUS,
            Planet.MERCURY: Sign.VIRGO,
            Planet.VENUS: Sign.PISCES,
            Planet.MARS: Sign.CAPRICORN,
            Planet.JUPITER: Sign.CANCER,
            Planet.SATURN: Sign.LIBRA,
        }

        # Traditional triplicity rulers (day/night)
        self.triplicity_rulers = {
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
            Sign.CANCER: {"day": Planet.MARS, "night": Planet.VENUS},
            Sign.SCORPIO: {"day": Planet.MARS, "night": Planet.VENUS},
            Sign.PISCES: {"day": Planet.MARS, "night": Planet.VENUS},
        }

    def calculate_comprehensive_reception(
        self, chart: HoraryChart, planet1: Planet, planet2: Planet
    ) -> Dict[str, Any]:
        """SINGLE SOURCE OF TRUTH for all reception calculations
        Returns comprehensive reception data used by both reasoning and structured output."""

        # Get planet positions
        pos1 = chart.planets[planet1]
        pos2 = chart.planets[planet2]

        # Determine day/night for triplicity calculations
        sun_pos = chart.planets[Planet.SUN]
        sun_house = self._calculate_house_position(sun_pos.longitude, chart.houses)
        is_day = sun_house in [7, 8, 9, 10, 11, 12]  # Sun below horizon = day chart

        # Check all dignity types for both directions
        reception_1_to_2 = self._check_all_dignities(planet1, pos2, is_day)
        reception_2_to_1 = self._check_all_dignities(planet2, pos1, is_day)

        # Determine overall reception type
        reception_type, reception_details = self._classify_reception(
            planet1, planet2, reception_1_to_2, reception_2_to_1
        )

        # Serialize one-way receptions for structured output
        one_way: List[str] = []
        for dignity in reception_1_to_2:
            label = "sign" if dignity == "domicile" else dignity
            one_way.append(f"{planet1.value}↦{planet2.value}({label})")
        for dignity in reception_2_to_1:
            label = "sign" if dignity == "domicile" else dignity
            one_way.append(f"{planet2.value}↦{planet1.value}({label})")

        # Only mutual receptions (including mixed) count for mutual field
        mutual_type = (
            reception_type
            if reception_type.startswith("mutual") or reception_type == "mixed_reception"
            else "none"
        )

        return {
            "type": reception_type,  # legacy field for existing logic
            "mutual": mutual_type,
            "one_way": one_way,
            "details": reception_details,
            "planet1_receives_planet2": reception_1_to_2,
            "planet2_receives_planet1": reception_2_to_1,
            "day_chart": is_day,
            "display_text": self._format_reception_display(
                reception_type, planet1, planet2, reception_details
            ),
            "traditional_strength": self._calculate_reception_strength(
                reception_1_to_2, reception_2_to_1
            ),
        }

    def _check_all_dignities(
        self, receiving_planet: Planet, received_position, is_day: bool
    ) -> List[str]:
        """Check all traditional dignity types for reception"""
        dignities: List[str] = []

        # 1. Domicile/Rulership (strongest)
        if received_position.sign.ruler == receiving_planet:
            dignities.append("domicile")

        # 2. Exaltation (second strongest)
        if (
            receiving_planet in self.exaltations
            and self.exaltations[receiving_planet] == received_position.sign
        ):
            dignities.append("exaltation")

        # 3. Triplicity (third strongest)
        if self._has_triplicity_dignity(
            receiving_planet, received_position.sign, is_day
        ):
            dignities.append("triplicity")

        # Get position within sign for terms/faces
        sign_degree = (
            received_position.longitude - received_position.sign.start_degree
        ) % 30

        # 4. Terms (Egyptian terms)
        try:
            terms_table = getattr(
                cfg().reception.terms, received_position.sign.sign_name
            )
            for term in terms_table:
                if term.start <= sign_degree < term.end:
                    if Planet[term.ruler.upper()] == receiving_planet:
                        dignities.append("term")
                    break
        except AttributeError:
            pass

        # 5. Faces/Decans
        try:
            faces_table = getattr(
                cfg().reception.faces, received_position.sign.sign_name
            )
            for face in faces_table:
                if face.start <= sign_degree < face.end:
                    if Planet[face.ruler.upper()] == receiving_planet:
                        dignities.append("face")
                    break
        except AttributeError:
            pass

        return dignities

    def _has_triplicity_dignity(self, planet: Planet, sign: Sign, is_day: bool) -> bool:
        """Check if planet has triplicity dignity in sign"""
        if sign not in self.triplicity_rulers:
            return False

        sect = "day" if is_day else "night"
        return self.triplicity_rulers[sign][sect] == planet

    def _classify_reception(
        self,
        planet1: Planet,
        planet2: Planet,
        reception_1_to_2: List[str],
        reception_2_to_1: List[str],
    ) -> Tuple[str, Dict]:
        """Classify the overall reception type"""

        # No reception
        if not reception_1_to_2 and not reception_2_to_1:
            return "none", {}

        # CRITICAL FIX: Prioritize mutual dignities by strength hierarchy
        # If both have domicile, it's mutual_rulership regardless of other dignities
        if "domicile" in reception_1_to_2 and "domicile" in reception_2_to_1:
            return "mutual_rulership", {
                "planet1_dignities": reception_1_to_2,
                "planet2_dignities": reception_2_to_1,
            }

        # If both have exaltation (and no domicile), it's mutual_exaltation
        if ("exaltation" in reception_1_to_2 and "exaltation" in reception_2_to_1 and
            "domicile" not in reception_1_to_2 and "domicile" not in reception_2_to_1):
            return "mutual_exaltation", {
                "planet1_dignities": reception_1_to_2,
                "planet2_dignities": reception_2_to_1,
            }

        # If both have terms (and no stronger dignities), it's mutual_term
        if ("term" in reception_1_to_2 and "term" in reception_2_to_1 and
            "domicile" not in reception_1_to_2 and "domicile" not in reception_2_to_1 and
            "exaltation" not in reception_1_to_2 and "exaltation" not in reception_2_to_1):
            return "mutual_term", {
                "planet1_dignities": reception_1_to_2,
                "planet2_dignities": reception_2_to_1,
            }

        # If both have faces (and no stronger dignities), it's mutual_face
        if ("face" in reception_1_to_2 and "face" in reception_2_to_1 and
            "domicile" not in reception_1_to_2 and "domicile" not in reception_2_to_1 and
            "exaltation" not in reception_1_to_2 and "exaltation" not in reception_2_to_1 and
            "term" not in reception_1_to_2 and "term" not in reception_2_to_1):
            return "mutual_face", {
                "planet1_dignities": reception_1_to_2,
                "planet2_dignities": reception_2_to_1,
            }

        # Mixed mutual reception - different dignity types
        if reception_1_to_2 and reception_2_to_1:
            return "mixed_reception", {
                "planet1_dignities": reception_1_to_2,
                "planet2_dignities": reception_2_to_1,
            }

        # Unilateral reception - one way only
        if reception_1_to_2:
            return "unilateral", {
                "receiving_planet": planet1,
                "received_planet": planet2,
                "dignities": reception_1_to_2,
            }
        else:
            return "unilateral", {
                "receiving_planet": planet2,
                "received_planet": planet1,
                "dignities": reception_2_to_1,
            }

    def _format_reception_display(
        self, reception_type: str, planet1: Planet, planet2: Planet, details: Dict
    ) -> str:
        """Format reception for display in reasoning"""
        if reception_type == "none":
            return "no reception"
        if reception_type == "mutual_rulership":
            return f"{planet1.value}↔{planet2.value} mutual domicile reception"
        if reception_type == "mutual_exaltation":
            return f"{planet1.value}↔{planet2.value} mutual exaltation reception"
        if reception_type == "mixed_reception":
            p1_dignities = ", ".join(details.get("planet1_dignities", []))
            p2_dignities = ", ".join(details.get("planet2_dignities", []))
            return (
                f"{planet1.value}↔{planet2.value} mixed reception ({p1_dignities} / {p2_dignities})"
            )
        if reception_type == "mutual_term":
            return f"{planet1.value}↔{planet2.value} mutual term reception"
        if reception_type == "mutual_face":
            return f"{planet1.value}↔{planet2.value} mutual face reception"
        if reception_type == "unilateral":
            receiving = details.get("receiving_planet")
            received = details.get("received_planet")
            dignities = ", ".join(details.get("dignities", []))
            return f"{receiving.value} receives {received.value} by {dignities}"
        return f"{reception_type} reception"

    def _calculate_reception_strength(
        self, reception_1_to_2: List[str], reception_2_to_1: List[str]
    ) -> int:
        """Calculate numerical reception strength considering both directions.

        Each dignity type maps to a base strength:
        domicile=5, exaltation=4, triplicity=3, term=2, face=1.
        The mutual reception strength is the minimum of the two directional
        strengths, with +1 synergy if either direction meets at least
        triplicity strength.
        """

        def strength_from_dignities(dignities: List[str]) -> int:
            if "domicile" in dignities:
                return 5
            if "exaltation" in dignities:
                return 4
            if "triplicity" in dignities:
                return 3
            if "term" in dignities:
                return 2
            if "face" in dignities:
                return 1
            return 0

        strength_a = strength_from_dignities(reception_1_to_2)
        strength_b = strength_from_dignities(reception_2_to_1)

        base_strength = min(strength_a, strength_b)
        synergy = 1 if (strength_a >= 3 or strength_b >= 3) else 0

        return base_strength + synergy

    def does_planet_receive(
        self, chart: HoraryChart, receiving_planet: Planet, received_planet: Planet
    ) -> Dict[str, Any]:
        """Check if receiving_planet has dignified reception of received_planet.
        
        Returns specific directional reception information to eliminate confusion
        about parameter order in comprehensive reception calculations.
        """
        reception_data = self.calculate_comprehensive_reception(chart, receiving_planet, received_planet)
        
        # Since we called calculate_comprehensive_reception(chart, receiving_planet, received_planet),
        # receiving_planet is always planet1, so we want planet1_receives_planet2
        dignities = reception_data["planet1_receives_planet2"]
            
        has_reception = len(dignities) > 0
        
        return {
            "has_reception": has_reception,
            "dignities": dignities,
            "reception_strength": self._calculate_single_direction_strength(dignities),
            "display_text": self._format_single_direction_display(receiving_planet, received_planet, dignities) if has_reception else "no reception"
        }
        
    def get_mutual_reception_only(
        self, chart: HoraryChart, planet1: Planet, planet2: Planet
    ) -> Dict[str, Any]:
        """Get only mutual reception data, filtering out unilateral receptions.
        
        This helper method makes it clearer when code specifically wants mutual reception
        and avoids confusion about checking individual directions.
        """
        comprehensive = self.calculate_comprehensive_reception(chart, planet1, planet2)
        
        is_mutual = (comprehensive["type"].startswith("mutual") or 
                    comprehensive["type"] == "mixed_reception")
        
        if not is_mutual:
            return {
                "has_mutual_reception": False,
                "mutual_type": "none",
                "mutual_strength": 0,
                "display_text": "no mutual reception"
            }
            
        return {
            "has_mutual_reception": True,
            "mutual_type": comprehensive["type"],
            "mutual_strength": comprehensive["traditional_strength"],
            "planet1_dignities": comprehensive["planet1_receives_planet2"],
            "planet2_dignities": comprehensive["planet2_receives_planet1"],
            "display_text": comprehensive["display_text"]
        }
        
    def _calculate_single_direction_strength(self, dignities: List[str]) -> int:
        """Calculate strength for a single direction of reception"""
        if "domicile" in dignities:
            return 5
        if "exaltation" in dignities:
            return 4
        if "triplicity" in dignities:
            return 3
        if "term" in dignities:
            return 2
        if "face" in dignities:
            return 1
        return 0
        
    def _format_single_direction_display(self, receiving: Planet, received: Planet, dignities: List[str]) -> str:
        """Format display text for single-direction reception"""
        if not dignities:
            return "no reception"
        dignity_text = ", ".join(dignities)
        dignity_label = "sign" if "domicile" in dignities else dignity_text
        return f"{receiving.value} receives {received.value} by {dignity_label}"

    def _calculate_house_position(self, longitude: float, houses: List[float]) -> int:
        """Helper method for house calculation"""
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
