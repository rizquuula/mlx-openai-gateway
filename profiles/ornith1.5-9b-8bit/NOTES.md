# Ornith-1.5-9B, 8-bit

Profile `ornith1.5-9b-8bit`. It serves `models/Ornith-1.5-9B-MLX-8bit` from
disk.

```bash
./serve-host.sh ornith1.5-9b-8bit
```

Fetch the weights first. The build is 9.54 GB in eight files:

```bash
./fetch-model.sh ornith1.5-9b-8bit
```

The model is a 9B text-only build of the `qwen3_5` architecture
(`Qwen3_5ForConditionalGeneration`). It reads no images and no video. The
quantization is affine, group size 64, 8 bits. `model_max_length` is 262144
tokens, and the card says YaRN rope scaling at factor 4.0 reaches about 1M
tokens.

It is a reasoning model. It emits `<think>` blocks and it calls tools. This
stack turns thinking off by default; set `MLX_THINKING=on` to restore it. The
card recommends `temperature=0.6` with `top_p=0.95`.

The natural comparison is the same model at 4-bit:
[ornith1.5-9b-4bit](../ornith1.5-9b-4bit/NOTES.md). It is the same weights at
half the size, so the pair isolates what the extra bits buy.

Read the [performance log preamble](../../README.md#performance-log) for the
hardware and for what TTFT and TPS mean here.

Benchmark this profile with:

```bash
MLX_PROFILE=ornith1.5-9b-8bit uv run --with openai python core/bench.py "ornith 8-bit"
```

## Performance

`models/Ornith-1.5-9B-MLX-8bit`.

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-21 | engine lm, thinking off | 0.69s | 3.43s | **14.6** | **52.4** |

Single-stream detail:

| Workload | 8-bit | 4-bit | 4-bit gain |
|---|---|---|---|
| short prompt | 14.6 | 25.7 | +76% |
| long generation | 14.7 | 22.3 | +52% |
| code generation | 13.9 | 26.6 | +91% |
| 2k-token prompt | 14.7 | 26.1 | +78% |

## What the numbers say

The 8-bit build decodes at **14.6 tok/s**. The [4-bit
build](../ornith1.5-9b-4bit/NOTES.md#performance) decodes at 25.9 tok/s on the
same machine, so 4-bit is **1.77x faster** for one stream.

That ratio is not a coincidence. The weights are 9.54 GB against 5.06 GB, a
factor of 1.89. Decode reads every weight once per token, so it is bound by
memory bandwidth, and speed tracks weight bytes almost exactly. Buying 8-bit
precision costs you close to half your tokens per second.

Prefill behaves differently. The 2k-token prompt costs 3.43s here and 3.18s at
4-bit, a difference of only 7%. Prefill multiplies whole matrices and is bound
by compute, not by bandwidth, so the extra bits are nearly free before the
first token.

Generation speed is flat across output length: 14.6, 14.7, 13.9, and 14.7 tok/s
across the four workloads. Nothing degrades as the answer grows.

Under 4 concurrent requests, aggregate throughput reaches 52.4 tok/s against
79.2 at 4-bit. The gap narrows from 77% to 51%, because batched decode reuses
each loaded weight across the requests in the batch and recovers part of the
bandwidth cost.

Pick 8-bit when output quality matters more than latency. Pick 4-bit for
interactive use. This benchmark measures speed only and says nothing about
which build answers better.

Untested so far: `--prefill-step-size` against that 3.43s prefill, and
speculative decoding through `--draft-model`.

