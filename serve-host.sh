#!/usr/bin/env bash
# Start the MLX engine on the host, where it can reach Metal.
# Containers cannot use Metal, so this process must stay outside Docker.
#
# mlx_vlm.server serves text AND images from one model load, so it replaces
# the text-only mlx_lm server entirely.
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

MODEL="${MLX_MODEL:-models/Qwen3.8-27B-Uncensored-MLX}"
PORT="${MLX_PORT:-8080}"
# Loopback only. OrbStack still routes host.docker.internal here, so the
# gateway container reaches the engine while the LAN cannot.
BIND="${MLX_BIND:-127.0.0.1}"

# The key is optional. Empty means no auth, which loopback makes reasonable.
auth_args=()
if [[ -n "${MLX_ENGINE_API_KEY:-}" ]]; then
  auth_args=(--api-key "${MLX_ENGINE_API_KEY}")
fi

if [[ "${BIND}" != "127.0.0.1" && -z "${MLX_ENGINE_API_KEY:-}" ]]; then
  echo "WARNING: engine binds ${BIND} with no API key." >&2
  echo "         Anyone on your network can use it. Set MLX_ENGINE_API_KEY," >&2
  echo "         or leave MLX_BIND at 127.0.0.1." >&2
fi

# This model reasons before answering, which costs hundreds of tokens even on
# trivial prompts. Set MLX_THINKING=on to restore it.
thinking_args=()
if [[ "${MLX_THINKING:-off}" == "on" ]]; then
  thinking_args=(--enable-thinking)
fi

# Speculative decoding. Off by default: it speeds up structured output such as
# code, but slows prose and collapses concurrent throughput. See README.
draft_args=()
if [[ -n "${MLX_DRAFT_MODEL:-}" ]]; then
  draft_args=(--draft-model "${MLX_DRAFT_MODEL}" --draft-kind "${MLX_DRAFT_KIND:-dflash}")
  echo "NOTE: speculative decoding is on. Expect faster code output," >&2
  echo "      slower prose, and much lower throughput under concurrency." >&2
fi

# Context window. The model itself accepts 262144 tokens, but memory sets the
# real ceiling: the KV cache grows 64 KiB per token at bf16. Quantizing the
# cache to 8 bits halves that and costs little accuracy. See README.
kv_args=()
if [[ -n "${MLX_MAX_KV_SIZE:-}" ]]; then
  kv_args+=(--max-kv-size "${MLX_MAX_KV_SIZE}")
fi
if [[ -n "${MLX_KV_BITS:-}" ]]; then
  kv_args+=(--kv-bits "${MLX_KV_BITS}")
fi

# Bound the continuous batch. One sequence keeps peak memory predictable, which
# is what makes a long context fit. Extra requests wait in the queue.
seq_args=()
if [[ -n "${MLX_MAX_NUM_SEQS:-}" ]]; then
  seq_args+=(--max-num-seqs "${MLX_MAX_NUM_SEQS}")
fi

auth_state=$([[ ${#auth_args[@]} -gt 0 ]] && echo on || echo off)
echo "Starting MLX engine: model=${MODEL} bind=${BIND}:${PORT} thinking=${MLX_THINKING:-off} auth=${auth_state}"
echo "  context=${MLX_MAX_KV_SIZE:-unbounded} kv_bits=${MLX_KV_BITS:-16} max_seqs=${MLX_MAX_NUM_SEQS:-unbounded}"
exec ./.venv/bin/python -m mlx_vlm.server \
  --model "${MODEL}" \
  --host "${BIND}" \
  --port "${PORT}" \
  --log-level INFO \
  ${auth_args[@]+"${auth_args[@]}"} \
  ${kv_args[@]+"${kv_args[@]}"} \
  ${seq_args[@]+"${seq_args[@]}"} \
  ${draft_args[@]+"${draft_args[@]}"} \
  ${thinking_args[@]+"${thinking_args[@]}"}
