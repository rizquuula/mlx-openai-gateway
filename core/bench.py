"""Measure engine performance: TTFT, single-stream TPS, and throughput TPS.

Run against the engine directly (port 8080) or through the gateway (8000):

    uv run --with openai python core/bench.py "baseline"
    uv run --with openai python core/bench.py "dflash" --base-url http://127.0.0.1:8000/v1

Print a markdown row ready to paste into the performance log in README.md.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent


def _profile_model() -> str | None:
    """Read MLX_MODEL out of the selected profile's model.env."""
    name = os.environ.get("MLX_PROFILE") or ""
    if not name:
        default = ROOT / "profiles" / "default"
        if default.is_file():
            name = default.read_text(encoding="utf-8").strip()
    if not name:
        return None
    model_env = ROOT / "profiles" / name / "model.env"
    if not model_env.is_file():
        return None
    # model.env is plain KEY=value, so split on the first "=" and nothing more.
    for line in model_env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "MLX_MODEL" and value.strip():
            return value.strip()
    return None


def _active_model() -> str:
    """Return the model id to send, mirroring the shell precedence.

    Do not read /v1/models. The engine hot-swaps: --model is only a preload, it
    loads whatever id a request names, and /v1/models lists every model it has
    seen. A wrong id here unloads the served model and downloads another.
    """
    explicit = os.environ.get("BENCH_MODEL") or os.environ.get("MLX_MODEL")
    if explicit:
        return explicit
    model = _profile_model()
    if model is None:
        raise SystemExit(
            "No model id. Set BENCH_MODEL or MLX_MODEL, or give the selected "
            "profile an MLX_MODEL line in its model.env."
        )
    return model


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


def stream_once(client: OpenAI, model: str, prompt: str, max_tokens: int) -> Run:
    """Stream one completion and time it."""
    start = time.time()
    ttft = 0.0
    tokens = 0
    stream = client.chat.completions.create(
        model=model,
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


def measure_single(client: OpenAI, model: str) -> list[tuple[str, Run]]:
    """Run each case once, one at a time."""
    return [
        (label, stream_once(client, model, prompt, n)) for label, prompt, n in CASES
    ]


def measure_throughput(client: OpenAI, model: str, concurrency: int) -> float:
    """Return aggregate tokens per second across concurrent requests."""
    prompt = "Write a detailed essay about distributed systems."
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        runs = list(
            pool.map(
                lambda _: stream_once(client, model, prompt, 200), range(concurrency)
            )
        )
    wall = time.time() - start
    return sum(r.tokens for r in runs) / wall if wall > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="name for this configuration, e.g. 'baseline'")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()

    model = _active_model()
    client = OpenAI(base_url=args.base_url, api_key="unused")

    print(f"== {args.label} == {model}")
    results = measure_single(client, model)
    for label, run in results:
        print(
            f"  {label:18} TTFT {run.ttft:5.2f}s  "
            f"{run.tps:5.1f} tok/s  ({run.tokens} toks)"
        )

    short_ttft = results[0][1].ttft
    long_ttft = results[-1][1].ttft
    single_tps = statistics.median(run.tps for _, run in results)
    throughput = measure_throughput(client, model, args.concurrency)
    print(f"  throughput x{args.concurrency:<7} {throughput:5.1f} tok/s aggregate")

    print("\nmarkdown row:")
    print(
        f"| {args.label} | {short_ttft:.2f}s | {long_ttft:.2f}s | "
        f"{single_tps:.1f} | {throughput:.1f} |"
    )


if __name__ == "__main__":
    main()
