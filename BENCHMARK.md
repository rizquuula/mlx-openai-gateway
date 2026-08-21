# Benchmark

Every profile on one page, measured on one machine.

Each model's own `NOTES.md` stays the source of truth. This file compiles the
tables so you can compare profiles, and links back for the detail and the
caveats.

Hardware: Apple M5, 10 GPU cores, 24 GB unified memory, macOS 26.6.1.
Stack: mlx 0.32.0, mlx-vlm 0.6.13, mlx-lm 0.31.3.

All rows ran with thinking off. Every row used the default wired limit except
the 27B speed row, which needs a raised one to finish at all.

## Speed

**TTFT** is time to first token. **Single TPS** is decode speed for one stream,
median across four workloads. **Throughput TPS** is aggregate tokens per second
across 4 concurrent requests.

| Profile | Quant | Engine | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|---|
| [qwen3.5-9b](profiles/qwen3.5-9b/NOTES.md#performance) | 4-bit | vlm | 0.35s | 3.13s | **27.2** | 73.8 |
| [ornith1.5-9b-4bit](profiles/ornith1.5-9b-4bit/NOTES.md#performance) | 4-bit | lm | 0.49s | 3.18s | **25.9** | 79.2 |
| [ornith1.5-9b-8bit](profiles/ornith1.5-9b-8bit/NOTES.md#performance) | 8-bit | lm | 0.69s | 3.43s | **14.6** | 52.4 |
| [qwen3.8-27b-uncensored](profiles/qwen3.8-27b-uncensored/NOTES.md#performance) | 4-bit | vlm | 1.32s | 11.51s | **8.4** | 22.0 |

The 27B row is the only one not measured on 2026-08-21. Its benchmark needs a
raised wired limit, and this round set none, so its figures come from the
earlier `wired 21504` run. See [what the 27B costs](#the-27b-needs-a-raised-wired-limit).

## Memory

Physical footprint from `vmmap -summary`. Do not use `ps` for this: MLX holds
the weights in Metal buffers in unified memory, and `ps rss` does not count
them. It reported 1.07 GiB for an engine holding 15 GiB.

All figures are GiB, the unit `vmmap` and `du` both report. Model cards quote
decimal GB, so the disk column below is smaller than the card's number for the
same files.

| Profile | Weights on disk | Resident after load | Peak under benchmark | Runtime above weights | Growth under load |
|---|---|---|---|---|---|
| [ornith1.5-9b-4bit](profiles/ornith1.5-9b-4bit/NOTES.md#memory) | 4.73 | 5.4 | 7.3 | 0.7 | 1.9 |
| [qwen3.5-9b](profiles/qwen3.5-9b/NOTES.md) | 5.57 | 6.7 | 8.5 | 1.1 | 1.8 |
| [ornith1.5-9b-8bit](profiles/ornith1.5-9b-8bit/NOTES.md#memory) | 8.90 | 9.6 | 11.4 | 0.7 | 1.8 |
| [qwen3.8-27b-uncensored](profiles/qwen3.8-27b-uncensored/NOTES.md#memory) | 15.01 | 17.6 | 20.5 | 2.6 | 2.9 |

## What the numbers say

### Quantization buys speed and memory at the same rate

The two Ornith profiles are the same model at two quantizations, so the pair
isolates what the extra bits cost.

| | 8-bit | 4-bit | 4-bit advantage |
|---|---|---|---|
| Single TPS | 14.6 | 25.9 | +77% |
| Resident after load | 9.6 GiB | 5.4 GiB | −4.2 GiB |
| Peak | 11.4 GiB | 7.3 GiB | −4.1 GiB |
| TTFT, 2k prompt | 3.43s | 3.18s | −7% |

Decode reads every weight once per token, so it is bound by memory bandwidth.
The weights differ by a factor of 1.89 and the speed differs by 1.77. Speed
tracks weight bytes almost exactly.

### Memory overhead is a constant, so weights decide the total

Two overheads sit above the weight file, and neither responds to quantization.
Loading costs about 0.7 GiB on a 9B. Serving the benchmark adds another 1.8 to
1.9 GiB, which is the KV cache and the activations. Those do not shrink when
the weights do.

So the memory difference between two builds of one model is just the weight
difference: 4.2 GiB resident against 4.17 GiB on disk for the Ornith pair.

To size a machine, add about 2.5 GiB to the weight file for a 9B and about
5.5 GiB for the 27B.

### Prefill barely responds to quantization

The 2k-token prompt costs 3.43s at 8-bit and 3.18s at 4-bit, a difference of
7%, against 77% for decode. Prefill multiplies whole matrices and is bound by
compute, not by bandwidth. Quantization is a decode optimization.

### The 27B needs a raised wired limit

At the default wired limit the 27B reached 20.5 GiB and its benchmark failed
with `kIOGPUCommandBufferCallbackErrorOutOfMemory`. It is the only profile here
that does not fit comfortably. The other three peak between 7.3 and 11.4 GiB
and need no tuning.

### Throughput is noisy, single-stream speed is not

Repeat runs on the same build agree on decode and disagree on concurrency:

| Profile | Single TPS | Throughput TPS |
|---|---|---|
| ornith1.5-9b-8bit | 14.6, 14.7 | 52.4, 38.6 |
| ornith1.5-9b-4bit | 25.9, 23.0 | 79.2, 54.2 |

Compare concurrency figures only within one run, and repeat a run before you
trust a throughput difference.

## Reproduce

Serve a profile, warm it with one throwaway request, then benchmark it:

```bash
./serve-host.sh ornith1.5-9b-4bit &
MLX_PROFILE=ornith1.5-9b-4bit uv run --with openai python core/bench.py "ornith 4-bit"
```

`core/bench.py` reads the model id from the same profile the scripts do. Set
`BENCH_MODEL` to override it.

Measure a run cold and the first workload reports a much larger TTFT, because
the first request pages the weights in from disk. Warm the engine first.

For memory, read the footprint of the engine process while it serves:

```bash
vmmap -summary "$(pgrep -f 'mlx_(vlm|lm)\.server' | head -1)" | grep 'Physical footprint'
```

## What this does not measure

Speed and memory only. Nothing here says which model answers better.
