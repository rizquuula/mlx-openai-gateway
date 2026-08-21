# MLX OpenAI Gateway

An OpenAI-compatible `/v1` endpoint for a local MLX model on Apple silicon.
Serves text and images from one model load.

Which model you serve is a [profile](#profiles). Four ship with the repo:

| Profile | Model | Notes |
|---|---|---|
| `qwen3.8-27b-uncensored` | `orcarouter/Qwen3.8-27B-Uncensored-MLX` | [NOTES.md](profiles/qwen3.8-27b-uncensored/NOTES.md) |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-4bit` | [NOTES.md](profiles/qwen3.5-9b/NOTES.md) |
| `ornith1.5-9b-8bit` | `ornith-ai/Ornith-1.5-9B-MLX-8bit` | [NOTES.md](profiles/ornith1.5-9b-8bit/NOTES.md) |
| `ornith1.5-9b-4bit` | `ornith-ai/Ornith-1.5-9B-MLX-4bit` | [NOTES.md](profiles/ornith1.5-9b-4bit/NOTES.md) |

`qwen3.8-27b-uncensored` is the default.

> **That default build is abliterated.** Its refusal training is removed, so
> it answers prompts a stock model declines. It is served on loopback for local use. Read
> the [model card](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)
> before you expose it to anyone else.

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

When `MLX_ENGINE=lm`, `MLX_ENGINE_API_KEY` is unavailable: `mlx_lm.server` reads
no key and no `Authorization` header. `core/serve.sh` refuses to start if the key
is set, rather than ignore it and leave port 8080 unguarded.

## Start

```bash
cp .env.example .env

./fetch-model.sh              # 1. weights for the default profile, 16.08 GB
./serve-host.sh &             # 2. engine on the host
docker compose up -d          # 3. gateway in Docker
```

Name a profile to serve a different model. The argument works on both scripts:

```bash
./serve-host.sh qwen3.5-9b
```

First start is slow: the engine reads 16 GB of weights from disk.

## Use

Any OpenAI client works. Point `base_url` at `http://127.0.0.1:8000/v1`.

```python
from openai import OpenAI

# The SDK demands a non-empty string, but the gateway ignores it unless you
# set GATEWAY_API_KEYS.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
# Send the MLX_MODEL line from profiles/<name>/model.env, verbatim.
MODEL = "models/Qwen3.8-27B-Uncensored-MLX"

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

Send the id the profile names. The engine loads whatever id a request carries,
so a different one makes it unload the running model and fetch that one instead.
`/v1/models` lists every model the engine has seen, not the one it serves.

## API docs

Swagger UI: <http://127.0.0.1:8000/docs> — ReDoc at `/redoc`.

No key is needed. If you set `GATEWAY_API_KEYS`, click *Authorize* and paste a
value before using *Try it out*.

The spec comes from the engine, re-addressed to `/v1` on the way through by
`openapi_rewriter.py`. Every path left in it is live. Three are dropped because
the gateway does not forward them; the module docstring says which and why.

`/healthz` is the gateway's own health check and needs no key.

`/docs` and `/redoc` need the `vlm` engine. `mlx_lm.server` serves no
`/openapi.json`, so there is no spec to re-address under `MLX_ENGINE=lm`.

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
| `MLX_BIND` | `127.0.0.1` | Engine bind address. Loopback keeps it off your LAN. |
| `MLX_THINKING` | `off` | Set to `on` to restore chain-of-thought. |
| `MLX_ENGINE` | `vlm` | Which engine serves the model. `lm` serves a text-only build that carries no vision tower. |
| `MLX_PROFILE` | *(from `profiles/default`)* | Which profile to serve. See [Profiles](#profiles). |

`MLX_MODEL` and the tuning knobs (`MLX_MAX_KV_SIZE`, `MLX_KV_BITS`,
`MLX_MAX_NUM_SEQS`, `MLX_DRAFT_MODEL`, `MLX_DRAFT_KIND`) belong to a model, so
they live in `profiles/<name>/model.env` instead.

`MLX_PROFILE` in `.env` does nothing. The scripts pick the profile before they
read `.env`, so set it in the environment or pass it as an argument.

## Profiles

A profile is one folder under `profiles/`. It holds everything that is true of
one model and nothing that is true of the stack:

| File | Holds |
|---|---|
| `model.env` | `MLX_MODEL`, the download source, and the tuning knobs. Plain `KEY=value`. |
| `files.txt` | Exact filenames for `fetch-model.sh`. Empty means the engine downloads the model itself. |
| `NOTES.md` | What this model costs in memory and speed on this machine. |

`profiles/default` is one line naming the profile you get when you ask for
nothing.

The scripts resolve the profile first, then layer the environment. First match
wins:

1. the argument: `./serve-host.sh qwen3.5-9b`
2. `MLX_PROFILE` in the environment: `MLX_PROFILE=qwen3.5-9b ./serve-host.sh`
3. `profiles/default`

Values then layer, highest first: the real environment, `model.env`, `.env`.
So `MLX_KV_BITS=8 ./serve-host.sh` works for one run without editing a file,
and the profile you named this run beats an `MLX_MODEL` left behind in `.env`.

To add a model, copy a profile folder, edit `model.env`, and serve it:

```bash
cp -r profiles/qwen3.5-9b profiles/my-model
$EDITOR profiles/my-model/model.env
./serve-host.sh my-model
```

`core/` holds the code every profile shares. `./serve-host.sh` and
`./fetch-model.sh` at the root are wrappers that call into it.

## Performance log

Hardware: Apple M5, 10 GPU cores, 24 GB unified memory, macOS 26.6.1.
Stack: mlx 0.32.0, mlx-vlm 0.6.13.

Reproduce any row with `uv run --with openai python core/bench.py "<label>"`.
The benchmark reads the model id from the same profile the scripts do, so
`MLX_PROFILE=qwen3.5-9b` picks the 9B. Set `BENCH_MODEL` to override it.

**TTFT** is time to first token. **Single TPS** is decode speed for one stream,
median across the four workloads. **Throughput TPS** is aggregate tokens per
second across 4 concurrent requests.

[BENCHMARK.md](BENCHMARK.md) compiles every profile into one speed table and
one memory table, and says what the numbers mean.

The numbers themselves live with the model they describe:

- [Qwen3.8-27B-Uncensored, 4-bit](profiles/qwen3.8-27b-uncensored/NOTES.md#performance)
- [Qwen3.5-9B, 4-bit](profiles/qwen3.5-9b/NOTES.md#performance)
- [Ornith-1.5-9B, 8-bit](profiles/ornith1.5-9b-8bit/NOTES.md#performance)
- [Ornith-1.5-9B, 4-bit](profiles/ornith1.5-9b-4bit/NOTES.md#performance)

## Thinking is off by default

Both models reason before answering. On the 9B, "Say the alphabet A to J" left
on consumed 2983 tokens and never produced an answer; with thinking off the same
prompt cost 3 tokens. On the 27B that prompt costs 20 tokens with thinking off
and answers directly.

Set `MLX_THINKING=on` if you want reasoning back, and budget `max_tokens` in the
thousands when you do. At 8.4 tok/s a long reasoning trace on the 27B is
expensive: 2000 reasoning tokens is about four minutes before any answer.

With thinking on, the engine streams the chain-of-thought in a non-standard
`reasoning` delta field, not `content`. Clients reading only `delta.content`
see nothing until reasoning ends.

## Stop

```bash
docker compose down
pkill -f mlx_vlm.server
```
