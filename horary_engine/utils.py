"""Miscellaneous utility helpers for the horary engine."""
from __future__ import annotations

from typing import Union

from .polarity_weights import TestimonyKey


def token_to_string(token: Union[TestimonyKey, str]) -> str:
    """Return a stable string representation of a testimony key.

    Parameters
    ----------
    token:
        Either a ``TestimonyKey`` enum member or a raw string.  The function
        is forgiving and will simply ``str()`` any non-enum input, making it
        safe for logging or JSON serialisation.
    """
    if isinstance(token, TestimonyKey):
        return token.value
    return str(token)
