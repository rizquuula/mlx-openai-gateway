#!/usr/bin/env bash
# Start the MLX engine on the host, where it can reach Metal.
# Containers cannot use Metal, so this process must stay outside Docker.
#
# mlx_vlm.server serves text AND images from one model load, so it is the
# default engine. mlx_lm.server is the fallback for a text-only build whose
# checkpoint carries no vision tower. MLX_ENGINE picks between them.
set -euo pipefail

# Run from the repo root. MLX_MODEL stays relative, and the engine reports it
# verbatim at /v1/models, so an absolute path would change the id every client
# sends. The cwd is what makes the relative path resolve.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# Pick the profile and layer .env under it. Sourced, never executed; see the
# header of core/profile.sh for why.
source "${ROOT}/core/profile.sh" "${1:-}"

# No fallback. The profile owns the model, so a profile without MLX_MODEL is a
# broken profile and should fail here rather than load something else.
MODEL="${MLX_MODEL}"
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

# Which engine serves the model. mlx_vlm serves text and images from one load,
# so it stays the default. Some checkpoints declare a vision architecture but
# ship no vision tower; mlx_vlm builds the tower from defaults and aborts on the
# missing parameters, and mlx_lm serves those builds as text.
#
# The two engines take different flags. Add a knob to one path and decide
# explicitly what the other path does with it: error, warn, or translate.
ENGINE="${MLX_ENGINE:-vlm}"
if [[ "${ENGINE}" != "vlm" && "${ENGINE}" != "lm" ]]; then
  echo "ERROR: unknown MLX_ENGINE '${ENGINE}'." >&2
  echo "       Valid values: vlm, lm." >&2
  exit 1
fi

if [[ "${ENGINE}" == "lm" ]]; then
  # If MLX_ENGINE_API_KEY is set on the lm engine, stop here. mlx_lm.server
  # reads no key and no Authorization header, so the key would be ignored and
  # the port would stay open. Refuse rather than serve an unguarded port.
  if [[ -n "${MLX_ENGINE_API_KEY:-}" ]]; then
    echo "ERROR: MLX_ENGINE_API_KEY is set, but MLX_ENGINE=lm has no authentication." >&2
    echo "       mlx_lm.server would ignore the key and leave port ${PORT} open." >&2
    echo "       Clear MLX_ENGINE_API_KEY, or serve this model on MLX_ENGINE=vlm." >&2
    exit 1
  fi

  # The lm engine has no --max-kv-size, --kv-bits, or --max-num-seqs. Name each
  # ignored knob and keep going: these tune speed and memory, and none of them
  # guards the port.
  if [[ -n "${MLX_MAX_KV_SIZE:-}" ]]; then
    echo "WARNING: MLX_MAX_KV_SIZE is set, but engine lm ignores it." >&2
  fi
  if [[ -n "${MLX_KV_BITS:-}" ]]; then
    echo "WARNING: MLX_KV_BITS is set, but engine lm ignores it." >&2
  fi
  if [[ -n "${MLX_MAX_NUM_SEQS:-}" ]]; then
    echo "WARNING: MLX_MAX_NUM_SEQS is set, but engine lm ignores it." >&2
  fi

  # Speculative decoding. The lm engine takes the draft model but picks the
  # draft kind itself, so warn about MLX_DRAFT_KIND only when drafting is on.
  # Every profile here sets MLX_DRAFT_KIND even with drafting off.
  lm_draft_args=()
  if [[ -n "${MLX_DRAFT_MODEL:-}" ]]; then
    lm_draft_args=(--draft-model "${MLX_DRAFT_MODEL}")
    echo "NOTE: speculative decoding is on. Expect faster code output," >&2
    echo "      slower prose, and much lower throughput under concurrency." >&2
    if [[ -n "${MLX_DRAFT_KIND:-}" ]]; then
      echo "WARNING: MLX_DRAFT_KIND is set, but engine lm ignores it." >&2
    fi
  fi

  # The lm engine has no --enable-thinking. The chat template reads
  # enable_thinking, so pass the value in both states. A template that reasons
  # by default turns thinking back on when the flag is absent.
  if [[ "${MLX_THINKING:-off}" == "on" ]]; then
    lm_template_args='{"enable_thinking": true}'
  else
    lm_template_args='{"enable_thinking": false}'
  fi

  echo "Starting MLX engine: profile=${PROFILE} model=${MODEL} bind=${BIND}:${PORT} thinking=${MLX_THINKING:-off} auth=off engine=${ENGINE}"
  echo "  mlx_lm.server: no api key, no kv-cache knobs, no concurrency limit"
  exec ./.venv/bin/python -m mlx_lm.server \
    --model "${MODEL}" \
    --host "${BIND}" \
    --port "${PORT}" \
    --log-level INFO \
    --chat-template-args "${lm_template_args}" \
    ${lm_draft_args[@]+"${lm_draft_args[@]}"}
fi

# Below here the vlm engine runs.

# Models reason before answering, which costs hundreds of tokens even on
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
echo "Starting MLX engine: profile=${PROFILE} model=${MODEL} bind=${BIND}:${PORT} thinking=${MLX_THINKING:-off} auth=${auth_state} engine=${ENGINE}"
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
