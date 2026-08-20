"""Typed settings for the gateway, loaded once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

DEFAULT_UPSTREAM: Final = "http://host.docker.internal:8080"
DEFAULT_TIMEOUT_SECONDS: Final = 600.0
# Vision requests carry base64 images inline, so the cap is well above the
# text-only default an OpenAI-shaped API would otherwise need.
DEFAULT_MAX_BODY_BYTES: Final = 32 * 1024 * 1024


class ConfigError(RuntimeError):
    """Raised when the environment does not describe a startable gateway."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the gateway needs to run. Built once, never mutated."""

    upstream_url: str
    api_keys: frozenset[str]
    upstream_api_key: str
    timeout_seconds: float
    max_body_bytes: int

    @property
    def auth_enabled(self) -> bool:
        """True when clients must present a key."""
        return bool(self.api_keys)

    @classmethod
    def from_env(cls) -> Settings:
        """Read settings from the environment.

        An empty GATEWAY_API_KEYS disables client auth. That is safe only
        because both ports bind loopback; the caller logs a warning.
        """
        keys = _parse_keys(os.getenv("GATEWAY_API_KEYS", ""))
        upstream_key = os.getenv("MLX_ENGINE_API_KEY", "").strip()
        return cls(
            upstream_url=os.getenv("MLX_UPSTREAM_URL", DEFAULT_UPSTREAM).rstrip("/"),
            api_keys=keys,
            upstream_api_key=upstream_key,
            timeout_seconds=_parse_float("GATEWAY_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            max_body_bytes=_parse_int("GATEWAY_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES),
        )


def _parse_keys(raw: str) -> frozenset[str]:
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def _parse_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _parse_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
