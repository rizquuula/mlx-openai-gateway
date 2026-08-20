# MLX OpenAI Gateway

An OpenAI-compatible `/v1` endpoint for `orcarouter/Qwen3.8-27B-Uncensored-MLX`
at 4-bit on Apple silicon. Serves text and images from one model load.

The engine loads whatever `MLX_MODEL` names, so the smaller
`mlx-community/Qwen3.5-9B-4bit` still works. See [Configuration](#configuration).

> **This build is abliterated.** Its refusal training is removed, so it answers
> prompts a stock model declines. It is served on loopback for local use. Read
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

## Get the weights

```bash
./fetch-model.sh              # 16.08 GB into models/
```

Do not point `MLX_MODEL` at the bare repo id. The repo carries four
quantizations in one tree — 2, 4, 6 and 8-bit, 93.85 GB in total — and the
4-bit build is mirrored at the repo root. `mlx_vlm` resolves a repo id with the
pattern `*.safetensors`, and `fnmatch` lets `*` cross a `/`, so that pattern
matches every subfolder copy too. Loading by repo id downloads all 93.85 GB.

`fetch-model.sh` asks for the 14 root files by exact name instead, which pulls
the 4-bit build alone. It retries on a dropped connection and resumes from the
bytes already on disk.

## Memory

The 4-bit weights are 16.05 GB on a 24 GB machine. A request peaks at 18.55 GB
for text and 18.84 GB with an image.

`iogpu.wired_limit_mb` caps how much memory the GPU may *wire*, meaning hold
resident and non-pageable. It is a ceiling, not a reservation: while the engine
is stopped the setting costs nothing.

**The 27B does not need a raised limit to run.** It loads at the default cap and
serves a single stream at full speed. What the raised limit buys is concurrency.

| | Default (unset) | `21504` (21 GB) |
|---|---|---|
| Model loads | yes, peak 18.55 GB | yes, peak 18.55 GB |
| Single stream | 8.0 tok/s | 8.4 tok/s |
| 4 concurrent | **Metal OOM**, one request dies | 22.0 tok/s, all complete |
| When it breaks | the request fails, the machine is fine | the machine rebooted |

Both failure modes are measured, not predicted. At the default cap a 4-way load
returns `kIOGPUCommandBufferCallbackErrorOutOfMemory`. With the limit at 21 GB a
57k-token prompt took the whole machine down, because 18.55 GB of pinned memory
left the kernel nothing to reclaim.

So the limit decides **who loses** when memory runs out. Unset, the model yields
and everything else keeps working. Raised, the model refuses to yield and macOS
and your other apps get squeezed. If you run emulators, VMs or Xcode alongside
the engine, leave it unset.

To raise it anyway:

```bash
sudo sysctl iogpu.wired_limit_mb=21504     # 21 GB
```

The setting resets on reboot, so it needs rerunning each time.

One caveat: after a Metal OOM the engine keeps answering health checks but every
later request fails. Restart it — `pkill -f mlx_vlm.server` — rather than
assuming the model is fine because `/healthz` is green.

## Context limits

The model accepts 262144 tokens. This machine does not. Weights take 16.05 GB
of 24 GB and a short request already peaks at 18.55 GB, so a few GB are left for
the KV cache and the prefill workspace. Raising the wired limit does not create
headroom, it only decides whether the model may pin what it takes.

What is measured, and what is not:

| Context | Result |
|---|---|
| ~2K tokens | works, 2.8-11.5s to first token |
| ~57K tokens | **exhausted memory and restarted the machine** |
| between | untested |

The 57K attempt ran with 8-bit KV quantization and `--max-num-seqs 1`, the two
settings that should have made it fit. It still failed. Prefill decayed from
102 tok/s to 35 tok/s as the cache grew, free memory reached 0.06 GB, swap
filled to 1.9 GB, and the machine rebooted.

Raise the context in small steps and watch `peak_memory` in the response
`timings`. Do not jump straight to a large value.

Two findings worth knowing before you tune:

- `--max-kv-size` **preallocates**. On an identical two-token prompt, a cap of
  4096 peaked at 18.54 GB and a cap of 65536 peaked at 19.73 GB. The cap costs
  memory whether or not you use the context.
- Prefill, not decode, is the wall. It is 174 tok/s on a 2K prompt and decays
  with length. Even if a long context fit, a 32K prompt would approach the
  600s `GATEWAY_TIMEOUT_SECONDS`.

The engine exposes the knobs through `serve-host.sh`, all off by default:

| Variable | Effect |
|---|---|
| `MLX_MAX_KV_SIZE` | Cap the KV cache in tokens. Preallocates. |
| `MLX_KV_BITS` | Quantize the KV cache, e.g. `8`. Costs prefill speed. |
| `MLX_MAX_NUM_SEQS` | Bound concurrent sequences. `1` makes peak memory predictable. |

For a genuinely long context on this hardware, use the 9B instead. Its weights
are about a third the size, which is where the headroom comes from.

## Start

```bash
cp .env.example .env

./serve-host.sh &             # 1. engine on the host
docker compose up -d          # 2. gateway in Docker
```

First start is slow: the engine reads 16 GB of weights from disk.

## Use

Any OpenAI client works. Point `base_url` at `http://127.0.0.1:8000/v1`.

```python
from openai import OpenAI

# The SDK demands a non-empty string, but the gateway ignores it unless you
# set GATEWAY_API_KEYS.
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
# The engine reports whatever MLX_MODEL named, so the id is the path.
# Confirm with: curl -s http://127.0.0.1:8000/v1/models
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
| `MLX_MODEL` | `models/Qwen3.8-27B-Uncensored-MLX` | Model to load. A local directory or a Hugging Face repo id. |
| `MLX_BIND` | `127.0.0.1` | Engine bind address. Loopback keeps it off your LAN. |
| `MLX_THINKING` | `off` | Set to `on` to restore chain-of-thought. |
| `MLX_MAX_KV_SIZE` | *(empty)* | KV cache cap in tokens. Preallocates. See [Context limits](#context-limits). |
| `MLX_KV_BITS` | *(empty)* | Quantize the KV cache, e.g. `8`. Costs prefill speed. |
| `MLX_MAX_NUM_SEQS` | *(empty)* | Cap concurrent sequences. Unbounded by default. |
| `MLX_DRAFT_MODEL` | *(empty)* | Speculative drafter, e.g. `z-lab/Qwen3.5-9B-DFlash`. See the performance log. |
| `MLX_DRAFT_KIND` | `dflash` | Drafter family: `dflash`, `eagle3`, or `mtp`. |

## Performance log

Hardware: Apple M5, 10 GPU cores, 24 GB unified memory, macOS 26.6.1.
Stack: mlx 0.32.0, mlx-vlm 0.6.13.

Reproduce any row with `uv run --with openai python bench.py "<label>"`.
Set `BENCH_MODEL` when the engine serves a different model.

**TTFT** is time to first token. **Single TPS** is decode speed for one stream,
median across the four workloads. **Throughput TPS** is aggregate tokens per
second across 4 concurrent requests.

### Qwen3.8-27B-Uncensored, 4-bit

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-20 | wired 21504 | 1.32s | 11.51s | **8.4** | **22.0** |
| 2026-08-20 | wired default | 0.89s | 2.77s | **8.0** | OOM at x4 |

The wired limit does not change single-stream speed: 8.0 against 8.4 tok/s is
noise. It changes what happens under concurrency. At the default cap the 4-way
stage failed with `kIOGPUCommandBufferCallbackErrorOutOfMemory`; at 21 GB it
completed at 22.0 tok/s. See [Memory](#memory).

TTFT on the 2k prompt varied a lot between runs, 11.51s and 2.77s, on identical
settings apart from the wired limit. Single-stream decode was unchanged across
both, so treat the 2k TTFT as noisy rather than as a wired-limit effect.

Single-stream detail:

| Workload | TTFT | tok/s |
|---|---|---|
| short prompt | 1.32s | 8.5 |
| long generation | 0.53s | 8.4 |
| code generation | 0.58s | 8.3 |
| 2k-token prompt | 11.51s | 8.4 |

Decode holds 8.3-8.5 tok/s across every workload, so output length does not
change the rate. Prefill is the weak spot again, and worse than on the 9B: a
2k-token prompt costs 11.51s before the first token, about 174 tok/s.

Peak memory is 18.6 GB for text and 18.8 GB with an image. Both exceed the
default Metal wired limit, so [Memory](#memory) is not optional here.

Against the 9B on the same machine, the 27B decodes 3.3x slower and prefills
3.2x slower. The trade is quality and a removed refusal layer for speed. Keep
the 9B for latency-sensitive work.

Untested on this model: `--kv-bits` quantization, `--prefill-step-size` against
that 11.51s prefill, and `--max-num-seqs` for concurrency.

Drafters exist, such as `z-lab/Qwen3.8-27B-DFlash2`, but none is benchmarked
here. Weights already occupy 18.6 GB of the 21 GB wired limit, so a second model
has about 2 GB of headroom. Expect it not to fit.

### Qwen3.5-9B-4bit

`mlx-community/Qwen3.5-9B-4bit`. The 2026-08-17 rows predate the 27B and ran
before any wired limit was set.

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-17 | wired default | 0.43s | 3.67s | **27.4** | **62.5** |
| 2026-08-17 | + DFlash drafter | 0.63s | 4.34s | 21.7 | 8.5 |
| 2026-08-20 | wired 21504 | 0.43s | 3.10s | 25.9 | 73.8 |

The 9B is the control for the wired-limit question. Its weights are about 5 GB,
so it never approaches either cap, and the limit makes no difference: 25.9
against 27.4 tok/s is run-to-run noise. Only a model that crowds the cap, like
the 27B, cares about the setting.

Measure a run cold and you will see a much larger TTFT on the first workload,
37.79s in one case, because the first request pages the weights in from disk.
Warm the engine with one throwaway request before benchmarking.

Single-stream detail, baseline vs DFlash:

| Workload | baseline | DFlash | change |
|---|---|---|---|
| short prompt | 27.6 | 20.1 | -27% |
| long generation | 27.4 | 23.3 | -15% |
| code generation | 27.3 | **40.5** | **+48%** |
| 2k-token prompt | 26.0 | 14.9 | -43% |

### What the numbers say

Generation holds ~27 tok/s regardless of output length. Prefill is the weak
spot: a 2k-token prompt costs 3.67s before the first token, about 560 tok/s.

Speculative decoding via `z-lab/Qwen3.5-9B-DFlash` is **off by default** because
it is a trade, not an upgrade. It wins big on structured output, where the
drafter predicts well: code generation reached 40.5 tok/s across three repeats
(39.6, 41.5, 40.5). It loses on prose, where drafts get rejected and the work is
wasted.

Concurrency is where it fails hardest. Aggregate throughput fell from 62.5 to
8.5 tok/s, confirmed at 7.7 and 6.3 on repeat runs. Speculation competes with
batched decode for the same GPU, so every extra client makes it worse.

Turn it on only for a single-user, code-heavy workload:

```bash
MLX_DRAFT_MODEL=z-lab/Qwen3.5-9B-DFlash ./serve-host.sh
```

Untested so far: `--kv-bits` quantization, `--prefill-step-size` against that
3.67s prefill, and `--max-num-seqs` for concurrency.

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
