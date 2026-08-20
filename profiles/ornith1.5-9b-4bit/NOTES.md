# Ornith-1.5-9B, 4-bit

Profile `ornith1.5-9b-4bit`. It serves `models/Ornith-1.5-9B-MLX-4bit` from
disk.

```bash
./serve-host.sh ornith1.5-9b-4bit
```

Fetch the weights first. The build is 5.06 GB in seven files:

```bash
./fetch-model.sh ornith1.5-9b-4bit
```

The model is a 9B text-only build of the `qwen3_5` architecture
(`Qwen3_5ForConditionalGeneration`). It reads no images and no video. The
quantization is affine, group size 64, 4 bits. `model_max_length` is 262144
tokens, and the card says YaRN rope scaling at factor 4.0 reaches about 1M
tokens.

It is a reasoning model. It emits `<think>` blocks and it calls tools. This
stack turns thinking off by default; set `MLX_THINKING=on` to restore it. The
card recommends `temperature=0.6` with `top_p=0.95`.

The natural comparison is the same model at 8-bit:
[ornith1.5-9b-8bit](../ornith1.5-9b-8bit/NOTES.md). It is the same weights at
double the size, so the pair isolates what the extra bits buy.

Read the [performance log preamble](../../README.md#performance-log) for the
hardware and for what TTFT and TPS mean here.

Benchmark this profile with:

```bash
MLX_PROFILE=ornith1.5-9b-4bit uv run --with openai python core/bench.py "ornith 4-bit"
```

## Performance

`models/Ornith-1.5-9B-MLX-4bit`.

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-21 | engine lm, thinking off | 0.49s | 3.18s | **25.9** | **79.2** |

Single-stream detail:

| Workload | 4-bit | 8-bit | 4-bit gain |
|---|---|---|---|
| short prompt | 25.7 | 14.6 | +76% |
| long generation | 22.3 | 14.7 | +52% |
| code generation | 26.6 | 13.9 | +91% |
| 2k-token prompt | 26.1 | 14.7 | +78% |

## What the numbers say

The 4-bit build decodes at **25.9 tok/s**, against 14.6 tok/s
for the [8-bit build](../ornith1.5-9b-8bit/NOTES.md#performance). 4-bit is
**1.77x faster** for one stream.

The weights are 5.06 GB against 9.54 GB, a factor of 1.89. Decode reads every
weight once per token, so it is bound by memory bandwidth and speed tracks
weight bytes almost exactly. The measured 1.77 sits just under the 1.89 the
weights predict.

Prefill gains almost nothing: 3.18s against 3.43s on the 2k-token prompt, a
difference of 7%. Prefill multiplies whole matrices and is bound by compute
rather than by bandwidth, so quantization barely moves it.

Under 4 concurrent requests, aggregate throughput reaches 79.2 tok/s against
52.4 at 8-bit. The advantage falls from 77% to 51%, because batched decode
reuses each loaded weight across the requests in the batch and recovers part of
the bandwidth cost that 8-bit pays.

This build matches `qwen3.5-9b` almost exactly at 25.9 tok/s, which is the
expected result: same architecture, same parameter count, same quantization.

Per-workload speed varies more here than at 8-bit, from 22.3 to 26.6 tok/s.
Long generation is the slow end and code generation the fast end.

Untested so far: `--prefill-step-size` against that 3.18s prefill, and
speculative decoding through `--draft-model`.

