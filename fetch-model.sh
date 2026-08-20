#!/usr/bin/env bash
# Download the 4-bit MLX build of Qwen3.8-27B-Uncensored into models/.
#
# Why this script does not use `hf download`:
#
# 1. The repo ships four quantizations in one tree — 2, 4, 6 and 8-bit, 93.85 GB
#    in total — and mirrors the 4-bit build at the repo root. mlx_vlm resolves a
#    repo id with the pattern "*.safetensors", and fnmatch lets "*" cross a "/",
#    so that pattern matches every subfolder copy. Loading by repo id downloads
#    all 93.85 GB. This script names the 14 root files instead: 16.08 GB.
#
# 2. huggingface_hub picks a fresh temporary name on every attempt, so a dropped
#    connection restarts the shard from byte zero. On a link that resets every
#    few minutes a 5.3 GB shard never lands. curl resumes with a byte range.
set -euo pipefail

cd "$(dirname "$0")"

REPO="${MODEL_REPO:-orcarouter/Qwen3.8-27B-Uncensored-MLX}"
DEST="${MODEL_DIR:-models/Qwen3.8-27B-Uncensored-MLX}"
REVISION="${MODEL_REVISION:-main}"
ATTEMPTS="${MODEL_FETCH_ATTEMPTS:-100}"

FILES=(
  "README.md"
  "chat_template.jinja"
  "config.json"
  "generation_config.json"
  "model-00001-of-00003.safetensors"
  "model-00002-of-00003.safetensors"
  "model-00003-of-00003.safetensors"
  "model.safetensors.index.json"
  "preprocessor_config.json"
  "processor_config.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "video_preprocessor_config.json"
  "vocab.json"
)

mkdir -p "${DEST}"

# Size the remote file so the loop knows when the download is complete.
# curl follows the redirect to the CDN and reports the final Content-Length.
remote_size() {
  local url="$1"
  # The redirect to the CDN emits its own Content-Length, so take the last one.
  curl -sSLI "$url" 2>/dev/null \
    | tr -d '\r' \
    | awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }'
}

local_size() {
  local path="$1"
  [[ -f "$path" ]] && wc -c < "$path" | tr -d ' ' || echo 0
}

echo "Fetching 4-bit weights: ${REPO} -> ${DEST}"

for name in "${FILES[@]}"; do
  url="https://huggingface.co/${REPO}/resolve/${REVISION}/${name}"
  out="${DEST}/${name}"

  want="$(remote_size "$url")"
  if [[ "${want}" -eq 0 ]]; then
    echo "Cannot size ${name}. Check the network and the repo id." >&2
    exit 1
  fi

  have="$(local_size "$out")"
  if [[ "${have}" -eq "${want}" ]]; then
    echo "ok       ${name}"
    continue
  fi

  echo "fetching ${name} ($(( want / 1000000 )) MB)"

  # -C - resumes from the bytes already written, so a reset costs only the
  # unflushed tail. The loop re-enters until the file reaches its full size.
  for attempt in $(seq 1 "${ATTEMPTS}"); do
    curl -fSL -C - --retry 5 --retry-delay 3 --retry-all-errors \
      --connect-timeout 20 -o "$out" "$url" || true

    have="$(local_size "$out")"
    [[ "${have}" -eq "${want}" ]] && break

    echo "  resume ${name}: ${have}/${want} bytes (attempt ${attempt})" >&2
    sleep 3
  done

  if [[ "${have}" -ne "${want}" ]]; then
    echo "Failed to fetch ${name}: ${have}/${want} bytes." >&2
    exit 1
  fi
done

echo "Fetch complete: ${DEST}"
