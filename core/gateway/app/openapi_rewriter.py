"""Reshapes the engine's OpenAPI spec to match what the gateway forwards."""

from __future__ import annotations

from typing import Any, Final

# The engine mounts almost every route both bare and under /v1, but not these.
# /health and /unload are bare-only, so a /v1 copy would 404. The engine
# documents /responses/input_tokens without mounting it at all, under either
# prefix. The gateway drops all three rather than document a dead endpoint.
BARE_ONLY_PATHS: Final[frozenset[str]] = frozenset(
    {"/health", "/unload", "/responses/input_tokens"}
)

_BEARER_SCHEME: Final[dict[str, Any]] = {
    "type": "http",
    "scheme": "bearer",
}


class OpenApiRewriter:
    """Prefixes engine paths with /v1 so Swagger calls the gateway correctly."""

    def __init__(self, bare_only: frozenset[str] = BARE_ONLY_PATHS) -> None:
        self._bare_only = bare_only

    def rewrite(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of the spec addressed to the gateway.

        The gateway only forwards /v1/*, so every documented path gains that
        prefix unless it already has one.
        """
        rewritten = dict(spec)
        rewritten["paths"] = {
            self._prefixed(path): item
            for path, item in spec.get("paths", {}).items()
            if path not in self._bare_only
        }
        rewritten["info"] = {
            **spec.get("info", {}),
            "title": "MLX Gateway (via mlx-vlm)",
        }
        _attach_bearer_auth(rewritten)
        return rewritten

    def _prefixed(self, path: str) -> str:
        if path.startswith("/v1/") or path == "/v1":
            return path
        return f"/v1{path}"


def _attach_bearer_auth(spec: dict[str, Any]) -> None:
    """Declare bearer auth so Swagger UI shows an Authorize button."""
    components = dict(spec.get("components", {}))
    schemes = dict(components.get("securitySchemes", {}))
    schemes["GatewayApiKey"] = _BEARER_SCHEME
    components["securitySchemes"] = schemes
    spec["components"] = components
    spec["security"] = [{"GatewayApiKey": []}]
