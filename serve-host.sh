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

MODEL="${MLX_MODEL:-mlx-community/Qwen3.5-9B-4bit}"
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

auth_state=$([[ ${#auth_args[@]} -gt 0 ]] && echo on || echo off)
echo "Starting MLX engine: model=${MODEL} bind=${BIND}:${PORT} thinking=${MLX_THINKING:-off} auth=${auth_state}"
exec ./.venv/bin/python -m mlx_vlm.server \
  --model "${MODEL}" \
  --host "${BIND}" \
  --port "${PORT}" \
  --log-level INFO \
  ${auth_args[@]+"${auth_args[@]}"} \
  ${thinking_args[@]+"${thinking_args[@]}"}
