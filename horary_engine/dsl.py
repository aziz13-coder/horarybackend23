"""Domain-specific language primitives for horary computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

try:
    from ..models import Planet, Aspect as AspectType
except ImportError:  # pragma: no cover - fallback when executed as script
    from models import Planet, Aspect as AspectType


# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Role:
    """Simple identifier for a significator role."""

    name: str


# Standard roles used throughout the engine
L1 = Role("L1")  # Lord of the 1st house – the querent
LQ = Role("LQ")  # Lord of the quesited's house
Moon = Role("Moon")  # The Moon as a general co-significator
L10 = Role("L10")  # Optional: Lord of the 10th house (career, outcome)
L3 = Role("L3")  # Optional: Lord of the 3rd house (communication, siblings)

# An actor in a primitive can be an actual planet or a role placeholder
Actor = Union[Planet, Role]


# ---------------------------------------------------------------------------
# Primitive data classes and constructors
# ---------------------------------------------------------------------------


@dataclass
class Aspect:
    """Relationship between two actors."""

    actor1: Actor
    actor2: Actor
    aspect: AspectType
    applying: bool


def aspect(actor1: Actor, actor2: Actor, aspect: AspectType, applying: bool = True) -> Aspect:
    return Aspect(actor1, actor2, aspect, applying)


@dataclass
class Translation:
    """Translation of light through a third actor."""

    translator: Actor
    from_actor: Actor
    to_actor: Actor
    applying: bool
    aspect: AspectType = AspectType.CONJUNCTION
    reception: bool = False


def translation(
    translator: Actor,
    from_actor: Actor,
    to_actor: Actor,
    applying: bool,
    aspect: AspectType = AspectType.CONJUNCTION,
    reception: bool = False,
) -> Translation:
    return Translation(translator, from_actor, to_actor, applying, aspect, reception)


@dataclass
class Collection:
    """Collection of light by a slower actor."""

    collector: Actor
    actor1: Actor
    actor2: Actor
    applying: bool
    aspect: AspectType = AspectType.CONJUNCTION
    reception: bool = False


def collection(
    collector: Actor,
    actor1: Actor,
    actor2: Actor,
    applying: bool,
    aspect: AspectType = AspectType.CONJUNCTION,
    reception: bool = False,
) -> Collection:
    return Collection(collector, actor1, actor2, applying, aspect, reception)


@dataclass
class Prohibition:
    """Interfering aspect preventing perfection."""

    prohibitor: Actor
    significator: Actor
    aspect: Optional[AspectType] = None


def prohibition(prohibitor: Actor, significator: Actor, aspect: Optional[AspectType] = None) -> Prohibition:
    return Prohibition(prohibitor, significator, aspect)


@dataclass
class Refranation:
    """One actor refrains from completing an aspect."""

    refrainer: Actor
    other: Actor


def refranation(refrainer: Actor, other: Actor) -> Refranation:
    return Refranation(refrainer, other)


@dataclass
class Frustration:
    """Third actor perfects aspect before main significators."""

    frustrator: Actor
    from_actor: Actor
    to_actor: Actor


def frustration(frustrator: Actor, from_actor: Actor, to_actor: Actor) -> Frustration:
    return Frustration(frustrator, from_actor, to_actor)


@dataclass
class Abscission:
    """Cutting off a connection between actors."""

    abscissor: Actor
    from_actor: Actor
    to_actor: Actor


def abscission(abscissor: Actor, from_actor: Actor, to_actor: Actor) -> Abscission:
    return Abscission(abscissor, from_actor, to_actor)


@dataclass
class Reception:
    """One actor receives another in dignity."""

    receiver: Actor
    received: Actor
    dignity: str  # e.g. "mutual", "sign", "exaltation"


def reception(receiver: Actor, received: Actor, dignity: str) -> Reception:
    return Reception(receiver, received, dignity)


@dataclass
class EssentialDignity:
    """Essential dignity indicator for an actor.

    The ``score`` field traditionally stores a numeric value but in some
    contexts we want to surface qualitative tokens (e.g. ``"detriment"``).
    To keep the primitive flexible we allow either a ``float`` or a string
    descriptor which higher level components may interpret as a discrete
    debility marker.
    """

    actor: Actor
    score: Union[float, str]


def essential(actor: Actor, score: Union[float, str]) -> EssentialDignity:
    return EssentialDignity(actor, score)


@dataclass
class AccidentalDignity:
    """Accidental dignity indicator for an actor.

    Similar to :class:`EssentialDignity`, the ``score`` may be a number or a
    qualitative token such as ``"retro"`` to denote retrogradation.
    """

    actor: Actor
    score: Union[float, str]


def accidental(actor: Actor, score: Union[float, str]) -> AccidentalDignity:
    return AccidentalDignity(actor, score)


@dataclass
class MoonVoidOfCourse:
    """Status of the Moon's void-of-course condition."""

    is_voc: bool
    detail: Optional[str] = None


def moon_voc(is_voc: bool, detail: Optional[str] = None) -> MoonVoidOfCourse:
    return MoonVoidOfCourse(is_voc, detail)


@dataclass
class HousePlacement:
    """Placement of an actor within a house."""

    actor: Actor
    house: int


def house(actor: Actor, house: int) -> HousePlacement:
    return HousePlacement(actor, house)


def is_benefic(planet: Planet) -> bool:
    """Return True if planet is traditionally benefic."""

    return planet in (Planet.JUPITER, Planet.VENUS)


def is_malefic(planet: Planet) -> bool:
    """Return True if planet is traditionally malefic."""

    return planet in (Planet.MARS, Planet.SATURN)


@dataclass
class RoleImportance:
    """Importance weighting for a role."""

    role: Role
    importance: float


def role_importance(role: Role, importance: float) -> RoleImportance:
    return RoleImportance(role, importance)


__all__ = [
    "Role",
    "L1",
    "LQ",
    "Moon",
    "L10",
    "L3",
    "aspect",
    "translation",
    "collection",
    "prohibition",
    "refranation",
    "frustration",
    "abscission",
    "reception",
    "essential",
    "accidental",
    "moon_voc",
    "house",
    "is_benefic",
    "is_malefic",
    "role_importance",
]
