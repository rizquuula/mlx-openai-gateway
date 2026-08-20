# Qwen3.5-9B, 4-bit

Profile `qwen3.5-9b`. It serves `mlx-community/Qwen3.5-9B-4bit` by repo id.

```bash
./serve-host.sh qwen3.5-9b
```

There is nothing to fetch first. This repo ships one build, so `mlx_vlm`
resolves the bare id and `huggingface_hub` caches about 5 GB under
`~/.cache/huggingface` on the first serve.

The weights are about a third of the 27B, which is where the headroom for a
longer context comes from. Its refusal training is intact.

Read the [performance log preamble](../../README.md#performance-log) for the
hardware and for what TTFT and TPS mean here.

## Performance

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

## What the numbers say

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
MLX_DRAFT_MODEL=z-lab/Qwen3.5-9B-DFlash ./serve-host.sh qwen3.5-9b
```

Untested so far: `--kv-bits` quantization, `--prefill-step-size` against that
3.67s prefill, and `--max-num-seqs` for concurrency.
