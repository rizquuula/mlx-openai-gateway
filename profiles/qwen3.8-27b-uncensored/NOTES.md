# Qwen3.8-27B-Uncensored, 4-bit

Profile `qwen3.8-27b-uncensored`. It serves
`orcarouter/Qwen3.8-27B-Uncensored-MLX` from `models/`, and it is the default
in `profiles/default`.

```bash
./fetch-model.sh qwen3.8-27b-uncensored    # once
./serve-host.sh                            # or: ./serve-host.sh qwen3.8-27b-uncensored
```

Read the [performance log preamble](../../README.md#performance-log) for the
hardware and for what TTFT and TPS mean here.

## Get the weights

```bash
./fetch-model.sh qwen3.8-27b-uncensored     # 16.08 GB into models/
```

Do not point `MLX_MODEL` at the bare repo id. The repo carries four
quantizations in one tree — 2, 4, 6 and 8-bit, 93.85 GB in total — and the
4-bit build is mirrored at the repo root. `mlx_vlm` resolves a repo id with the
pattern `*.safetensors`, and `fnmatch` lets `*` cross a `/`, so that pattern
matches every subfolder copy too. Loading by repo id downloads all 93.85 GB.

`files.txt` in this folder names the 14 root files instead, which pulls the
4-bit build alone. `core/fetch.sh` retries on a dropped connection and resumes
from the bytes already on disk.

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

Set the knobs in `model.env` in this folder, or in the environment for one
run. All are off by default:

| Variable | Effect |
|---|---|
| `MLX_MAX_KV_SIZE` | Cap the KV cache in tokens. Preallocates. |
| `MLX_KV_BITS` | Quantize the KV cache, e.g. `8`. Costs prefill speed. |
| `MLX_MAX_NUM_SEQS` | Bound concurrent sequences. `1` makes peak memory predictable. |

For a genuinely long context on this hardware, use the 9B instead. Its weights
are about a third the size, which is where the headroom comes from.

## Performance

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
