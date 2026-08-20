"""Measure engine performance: TTFT, single-stream TPS, and throughput TPS.

Run against the engine directly (port 8080) or through the gateway (8000):

    uv run --with openai python bench.py "baseline"
    uv run --with openai python bench.py "dflash" --base-url http://127.0.0.1:8000/v1

Print a markdown row ready to paste into the performance log in README.md.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from openai import OpenAI

# The engine serves one model, so the id only has to match what it reports.
# Override when you benchmark a different load: BENCH_MODEL=... python bench.py
MODEL = os.environ.get("BENCH_MODEL", "models/Qwen3.8-27B-Uncensored-MLX")
LONG_PROMPT = (
    "Background notes.\n"
    + ("The system processes events in order. " * 300)
    + "\nSummarize the above in one sentence."
)
CASES: list[tuple[str, str, int]] = [
    ("short prompt", "Write a short paragraph about the ocean.", 200),
    ("long generation", "Write a detailed essay about distributed systems.", 400),
    ("code generation", "Write a Python function that merges two sorted lists.", 300),
    ("2k-token prompt", LONG_PROMPT, 100),
]
CONCURRENCY = 4


@dataclass(frozen=True, slots=True)
class Run:
    """One streamed completion, timed."""

    ttft: float
    tokens: int
    elapsed: float

    @property
    def tps(self) -> float:
        """Tokens per second during decode, excluding time to first token."""
        decode = self.elapsed - self.ttft
        return self.tokens / decode if decode > 0 else 0.0


def stream_once(client: OpenAI, prompt: str, max_tokens: int) -> Run:
    """Stream one completion and time it."""
    start = time.time()
    ttft = 0.0
    tokens = 0
    stream = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        stream=True,
        messages=[{"role": "user", "content": prompt}],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        if tokens == 0:
            ttft = time.time() - start
        tokens += 1
    return Run(ttft=ttft, tokens=tokens, elapsed=time.time() - start)


def measure_single(client: OpenAI) -> list[tuple[str, Run]]:
    """Run each case once, one at a time."""
    return [(label, stream_once(client, prompt, n)) for label, prompt, n in CASES]


def measure_throughput(client: OpenAI, concurrency: int) -> float:
    """Return aggregate tokens per second across concurrent requests."""
    prompt = "Write a detailed essay about distributed systems."
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(
            pool.map(lambda _: stream_once(client, prompt, 200), range(concurrency))
        )
    wall = time.time() - start
    return sum(r.tokens for r in runs) / wall if wall > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="name for this configuration, e.g. 'baseline'")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="unused")

    print(f"== {args.label} ==")
    results = measure_single(client)
    for label, run in results:
        print(
            f"  {label:18} TTFT {run.ttft:5.2f}s  "
            f"{run.tps:5.1f} tok/s  ({run.tokens} toks)"
        )

    short_ttft = results[0][1].ttft
    long_ttft = results[-1][1].ttft
    single_tps = statistics.median(run.tps for _, run in results)
    throughput = measure_throughput(client, args.concurrency)
    print(f"  throughput x{args.concurrency:<7} {throughput:5.1f} tok/s aggregate")

    print("\nmarkdown row:")
    print(
        f"| {args.label} | {short_ttft:.2f}s | {long_ttft:.2f}s | "
        f"{single_tps:.1f} | {throughput:.1f} |"
    )


if __name__ == "__main__":
    main()
