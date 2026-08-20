"""OpenAI-compatible gateway in front of a host-native MLX engine."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.authenticator import ApiKeyAuthenticator
from app.config import Settings
from app.openapi_rewriter import OpenApiRewriter
from app.upstream_proxy import (
    UpstreamProxy,
    UpstreamUnavailableError,
    response_headers,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gateway")

settings = Settings.from_env()
authenticator = ApiKeyAuthenticator(settings.api_keys)
_rewriter = OpenApiRewriter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the upstream HTTP client for the process lifetime."""
    timeout = httpx.Timeout(settings.timeout_seconds, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        app.state.proxy = UpstreamProxy(
            client, settings.upstream_url, settings.upstream_api_key
        )
        if not authenticator.enabled:
            logger.warning(
                "client auth is OFF. Anyone who can reach this port can use the "
                "model. Safe only while the port stays bound to loopback."
            )
        logger.info(
            "gateway ready, upstream=%s auth=%s",
            settings.upstream_url,
            "on" if authenticator.enabled else "off",
        )
        yield


# The gateway's own generated spec would only describe the proxy, so all three
# doc paths are freed and served from the engine instead.
app = FastAPI(
    title="MLX OpenAI Gateway",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

_SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff"}

# Documentation is intentionally readable without a key. The endpoints it
# describes still require one.
_DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness probe. Unauthenticated by design; reveals nothing."""
    return JSONResponse({"status": "ok"}, headers=_SECURITY_HEADERS)


@app.get("/docs", include_in_schema=False)
@app.get("/redoc", include_in_schema=False)
async def docs(request: Request) -> Response:
    """Serve the engine's Swagger and ReDoc pages, unauthenticated by request.

    Only the doc paths skip the key check. Everything under /v1 still
    requires one.
    """
    return await _forward(request, request.url.path)


@app.get("/openapi.json", include_in_schema=False)
async def openapi(request: Request) -> Response:
    """Serve the engine's spec, re-addressed to the gateway's /v1 prefix."""
    proxy: UpstreamProxy = request.app.state.proxy
    try:
        spec = await proxy.fetch_json("/openapi.json")
    except UpstreamUnavailableError as exc:
        logger.error("cannot read upstream spec: %s", exc)
        return _error(502, "upstream_unavailable", "The model backend is unavailable.")
    return JSONResponse(_rewriter.rewrite(spec), headers=_SECURITY_HEADERS)


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy_v1(path: str, request: Request) -> Response:
    """Authenticate, then forward everything under /v1 to the MLX engine."""
    if not authenticator.is_authorized(request.headers.get("authorization")):
        return _error(401, "invalid_api_key", "Incorrect API key provided.")
    return await _forward(request, f"/v1/{path}")


async def _forward(request: Request, path: str) -> Response:
    """Relay one request upstream and stream the answer back."""
    body = await request.body()
    if len(body) > settings.max_body_bytes:
        return _error(413, "request_too_large", "Request body exceeds the limit.")

    proxy: UpstreamProxy = request.app.state.proxy
    try:
        upstream, stream = await proxy.open_stream(
            method=request.method,
            path=path,
            headers=dict(request.headers),
            body=body,
            query=request.url.query,
        )
    except UpstreamUnavailableError as exc:
        logger.error("upstream failure path=%s error=%s", path, exc)
        return _error(502, "upstream_unavailable", "The model backend is unavailable.")

    logger.info("proxied %s %s -> %d", request.method, path, upstream.status_code)
    return StreamingResponse(
        _drain(upstream, stream),
        status_code=upstream.status_code,
        headers={**response_headers(upstream), **_SECURITY_HEADERS},
    )


async def _drain(
    upstream: httpx.Response, stream: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    """Relay upstream bytes, then always release the connection."""
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await upstream.aclose()


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Return an OpenAI-shaped error. Detail stays in the log, not the response."""
    return JSONResponse(
        {"error": {"message": message, "type": code, "code": code}},
        status_code=status,
        headers=_SECURITY_HEADERS,
    )
