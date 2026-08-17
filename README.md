# MLX OpenAI Gateway

An OpenAI-compatible `/v1` endpoint for `mlx-community/Qwen3.5-9B-4bit` on Apple
silicon. Serves text and images from one model load.

## Why the engine is not in the container

macOS does not pass Metal through to containers. Docker, OrbStack, and Apple's
own `container` tool all lack it. A containerized MLX runs on CPU and loses the
GPU entirely.

So the stack splits in two:

| Part | Where it runs | Why |
|---|---|---|
| MLX engine (`mlx_vlm.server`) | host, port 8080 | needs Metal |
| Gateway (`/v1`, auth, logging) | container, port 8000 | needs no GPU |

The gateway reaches the engine over `host.docker.internal`.

## Security model: loopback, no keys

Both ports bind `127.0.0.1` and neither is reachable from the LAN, so API keys
are off by default.

OrbStack routes `host.docker.internal` to macOS loopback services, which is what
lets the engine stay on `127.0.0.1` while the container still reaches it. On
Docker Desktop that usually does not work; there you must set `MLX_BIND=0.0.0.0`
and, because that exposes the engine to your network, set `MLX_ENGINE_API_KEY`
as well.

To turn auth back on, set either key in `.env`:

| Variable | Guards |
|---|---|
| `GATEWAY_API_KEYS` | port 8000, presented by your clients |
| `MLX_ENGINE_API_KEY` | port 8080, presented by the gateway |

Set one and that port starts requiring it. Leave it empty and that port is open.
A client key is never forwarded to the engine.

## Start

```bash
cp .env.example .env

./serve-host.sh &             # 1. engine on the host
docker compose up -d          # 2. gateway in Docker
```

## Use

Any OpenAI client works. Point `base_url` at `http://127.0.0.1:8000/v1`.

```python
from openai import OpenAI

# The SDK demands a non-empty string, but the gateway ignores it unless you
# set GATEWAY_API_KEYS.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
MODEL = "mlx-community/Qwen3.5-9B-4bit"

# text
client.chat.completions.create(
    model=MODEL, messages=[{"role": "user", "content": "Hello"}]
)

# image, standard OpenAI content parts
import base64
b64 = base64.b64encode(open("photo.png", "rb").read()).decode()
client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
]}])
```

Streaming works on both paths (`stream=True`).

## API docs

Swagger UI: <http://127.0.0.1:8000/docs> — ReDoc at `/redoc`.

No key is needed. If you set `GATEWAY_API_KEYS`, click *Authorize* and paste a
value before using *Try it out*.

The spec comes from the engine, rewritten on the way through. The engine
documents its routes bare (`/chat/completions`) but the gateway only forwards
`/v1/*`, so `openapi_rewriter.py` re-addresses every path to the `/v1` form the
gateway actually serves. Three documented routes are dropped because no `/v1`
form exists: `/health` and `/unload` are bare-only, and the engine documents
`/responses/input_tokens` without mounting it anywhere.

Every path left in the spec is live. `/health` has a gateway equivalent at
`/healthz`, which needs no key.

Swagger UI loads its CSS and JS from a CDN, so the page itself needs internet
even though the API does not.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GATEWAY_API_KEYS` | *(empty)* | Client keys, comma-separated. Empty disables auth. |
| `MLX_ENGINE_API_KEY` | *(empty)* | Key the gateway uses toward the engine. Empty disables it. |
| `GATEWAY_PORT` | `8000` | Gateway port, bound to `127.0.0.1`. |
| `MLX_UPSTREAM_URL` | `http://host.docker.internal:8080` | Where the engine listens. |
| `GATEWAY_TIMEOUT_SECONDS` | `600` | Upstream read timeout. |
| `GATEWAY_MAX_BODY_BYTES` | `33554432` | 32 MB. Inline base64 images are large. |
| `MLX_MODEL` | `mlx-community/Qwen3.5-9B-4bit` | Model to load. |
| `MLX_BIND` | `127.0.0.1` | Engine bind address. Loopback keeps it off your LAN. |
| `MLX_THINKING` | `off` | Set to `on` to restore chain-of-thought. |

## Thinking is off by default

This model reasons before answering. Left on, "Say the alphabet A to J" consumed
2983 tokens and never produced an answer. With thinking off the same prompt
costs 3 tokens. Set `MLX_THINKING=on` if you want it back, and budget
`max_tokens` in the thousands when you do.

With thinking on, the engine streams the chain-of-thought in a non-standard
`reasoning` delta field, not `content`. Clients reading only `delta.content`
see nothing until reasoning ends.

## Stop

```bash
docker compose down
pkill -f mlx_vlm.server
```
