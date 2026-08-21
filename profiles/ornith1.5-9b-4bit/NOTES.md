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

`models/Ornith-1.5-9B-MLX-4bit`. Both rows ran on the same day at the default
wired limit, back to back with the other profiles.

| Date | Config | TTFT short | TTFT 2k | Single TPS | Throughput TPS |
|---|---|---|---|---|---|
| 2026-08-21 | engine lm, thinking off | 0.49s | 3.18s | **25.9** | **79.2** |
| 2026-08-21 | repeat | 0.50s | 4.48s | **23.0** | **54.2** |

Single-stream detail, first run:

| Workload | TTFT | tok/s |
|---|---|---|
| short prompt | 0.49s | 25.7 |
| long generation | 0.38s | 22.3 |
| code generation | 0.38s | 26.6 |
| 2k-token prompt | 3.18s | 26.1 |

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

The 4-bit build decodes at **23.0 to 25.9 tok/s**, against 14.6 to 14.7 for the
[8-bit build](../ornith1.5-9b-8bit/NOTES.md#performance). 4-bit is **1.6 to
1.8x faster** for one stream.

The weights differ by a factor of 1.89, 4.73 GiB against 8.90 GiB. Decode reads
every weight once per token, so it is bound by memory bandwidth and speed
tracks weight bytes. The measured range sits just under what the weights
predict.

Prefill gains almost nothing: 3.18s against 3.43s on the 2k-token prompt in the
first run. Prefill multiplies whole matrices and is bound by compute rather than
by bandwidth, so quantization barely moves it.

Single-stream decode is the stable measurement here. Aggregate throughput is
not: this build gave 79.2 and 54.2 tok/s on two runs, and the 8-bit gave 52.4
and 38.6. Compare concurrency figures only within one run, and repeat a run
before you trust a throughput difference.

This build matches `qwen3.5-9b`, which measured 27.2 tok/s and 6.7 GB resident
on the same day. Same architecture, same parameter count, same quantization, so
the agreement is the expected result.

Per-workload speed varies more here than at 8-bit, from 22.3 to 26.6 tok/s.
Long generation is the slow end and code generation the fast end.

Untested so far: `--prefill-step-size` against that prefill, and speculative
decoding through `--draft-model`.
