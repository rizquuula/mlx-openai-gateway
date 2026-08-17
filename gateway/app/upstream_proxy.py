"""Streaming reverse proxy to the host-native mlx_lm server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Final

import httpx

# Hop-by-hop headers are per-connection and must not be forwarded.
_HOP_BY_HOP: Final[frozenset[str]] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "authorization",
    }
)


class UpstreamUnavailableError(RuntimeError):
    """Raised when the MLX engine cannot be reached or does not answer in time."""


class UpstreamProxy:
    """Forwards a request to the MLX engine and streams the response back."""

    def __init__(
        self, client: httpx.AsyncClient, base_url: str, upstream_api_key: str
    ) -> None:
        self._client = client
        self._base_url = base_url
        self._upstream_api_key = upstream_api_key

    async def open_stream(
        self,
        *,
        method: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
        query: str,
    ) -> tuple[httpx.Response, AsyncIterator[bytes]]:
        """Send the request upstream and return the open response plus its byte stream.

        The caller owns closing the response.
        """
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"
        # The caller's key is stripped as hop-by-hop; the gateway presents its
        # own credential instead, when the engine asks for one.
        outbound = _forwardable(headers)
        if self._upstream_api_key:
            outbound["Authorization"] = f"Bearer {self._upstream_api_key}"
        request = self._client.build_request(
            method,
            url,
            headers=outbound,
            content=body or None,
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailableError("upstream timed out") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailableError("upstream unreachable") from exc
        return response, response.aiter_raw()


    async def fetch_json(self, path: str) -> dict:
        """GET one small JSON document from the engine and return it parsed."""
        auth = (
            {"Authorization": f"Bearer {self._upstream_api_key}"}
            if self._upstream_api_key
            else {}
        )
        try:
            response = await self._client.get(f"{self._base_url}{path}", headers=auth)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailableError(f"cannot read {path}") from exc
        except ValueError as exc:
            raise UpstreamUnavailableError(f"{path} is not valid JSON") from exc


def _forwardable(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def response_headers(response: httpx.Response) -> dict[str, str]:
    """Strip hop-by-hop headers from an upstream response."""
    return {
        k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP
    }
