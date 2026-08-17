"""Bearer-token check for inbound requests."""

from __future__ import annotations

import secrets
from collections.abc import Iterable

_BEARER_PREFIX = "Bearer "


class ApiKeyAuthenticator:
    """Verifies an Authorization header against a fixed set of API keys."""

    def __init__(self, api_keys: Iterable[str]) -> None:
        self._api_keys = frozenset(api_keys)

    @property
    def enabled(self) -> bool:
        """False when no keys are configured, which allows every caller."""
        return bool(self._api_keys)

    def is_authorized(self, authorization_header: str | None) -> bool:
        """Return True when the header carries a known key.

        With no keys configured, auth is off and every caller passes.
        Otherwise every candidate key is compared even after a match, so the
        running time does not reveal which key matched or how many exist.
        """
        if not self._api_keys:
            return True
        presented = _extract_token(authorization_header)
        if presented is None:
            return False
        matched = False
        for known in self._api_keys:
            if secrets.compare_digest(presented, known):
                matched = True
        return matched


def _extract_token(header: str | None) -> str | None:
    if header is None:
        return None
    if not header.startswith(_BEARER_PREFIX):
        return None
    token = header[len(_BEARER_PREFIX) :].strip()
    if not token:
        return None
    return token
