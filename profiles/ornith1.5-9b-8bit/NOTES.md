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

`models/Ornith-1.5-9B-MLX-8bit`. Both rows ran on the same day at the default
wired limit, back to back with the other profiles.

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-21 | engine lm, thinking off | 0.69s | 3.43s | **14.6** | **52.4** |
| 2026-08-21 | repeat | 0.61s | 4.83s | **14.7** | **38.6** |

Single-stream detail, first run:

| Workload | TTFT | tok/s |
|---|---|---|
| short prompt | 0.69s | 14.6 |
| long generation | 0.44s | 14.7 |
| code generation | 0.42s | 13.9 |
| 2k-token prompt | 3.43s | 14.7 |

## Memory

All figures are GiB, the unit `vmmap` and `du` both report.

| Stage | 8-bit | 4-bit | 8-bit costs |
|---|---|---|---|
| weights on disk | 8.90 | 4.73 | +4.17 |
| resident after load | 9.6 | 5.4 | +4.2 |
| peak under the benchmark | 11.4 | 7.3 | +4.1 |
| runtime above weights | 0.7 | 0.7 | none |
| growth under load | 1.8 | 1.9 | none |

Measured as the physical footprint reported by `vmmap -summary`, not by `ps`.
MLX holds the weights in Metal buffers in unified memory, and `ps rss` does not
count them: it reported 1.07 GiB for an engine holding 15 GiB.

Read the last two rows first. Loading costs about 0.7 GiB beyond the weight
files on both builds, and serving the benchmark adds another 1.8 to 1.9 GiB on
both. Neither figure responds to quantization, because the KV cache and the
activations do not shrink when the weights do.

So the whole memory difference between these two profiles is the weight
difference: 4.2 GiB resident, against 4.17 GiB on disk. Choosing 4-bit saves
that and nothing else.

Both builds fit this 24 GB machine with room to spare at the default wired
limit. Neither needs the tuning the 27B requires.

## What the numbers say

The 8-bit build decodes at **14.6 to 14.7 tok/s**, and the repeat run confirms
it. The [4-bit build](../ornith1.5-9b-4bit/NOTES.md#performance) reached 25.9
and 23.0 tok/s on the same two runs, so 4-bit decodes **1.6 to 1.8x faster**.

The weights differ by a factor of 1.89, 8.90 GiB against 4.73 GiB. Decode reads
every weight once per token, so it is bound by memory bandwidth and speed
tracks weight bytes. The measured range sits just under what the weights
predict.

Prefill gains far less. The 2k-token prompt cost 3.43s and 4.83s here against
3.18s and 4.48s at 4-bit. Prefill multiplies whole matrices and is bound by
compute rather than by bandwidth, so the extra bits are nearly free before the
first token.

Single-stream decode is the stable measurement here. Aggregate throughput is
not: the same 8-bit build gave 52.4 and 38.6 tok/s on two runs, and the 4-bit
gave 79.2 and 54.2. Compare concurrency figures only within one run, and repeat
a run before you trust a throughput difference.

Decode speed does not change with output length: 14.6, 14.7, 13.9, and 14.7
tok/s across the four workloads.

Pick 8-bit when output quality matters more than latency and you can spare the
4.2 GiB. Pick 4-bit for interactive use. This benchmark measures speed and
memory only. It says nothing about which build answers better.

Untested so far: `--prefill-step-size` against that prefill, and speculative
decoding through `--draft-model`.
