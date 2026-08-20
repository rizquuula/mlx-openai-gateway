#!/usr/bin/env bash
# Download a profile's weights into models/.
#
# Why this script does not use `hf download`: huggingface_hub picks a fresh
# temporary name on every attempt, so a dropped connection restarts the shard
# from byte zero. On a link that resets every few minutes a 5.3 GB shard never
# lands. curl resumes with a byte range.
#
# Which files to pull is the profile's business. See the header of each
# profiles/<name>/files.txt for why that list looks the way it does.
set -euo pipefail

# Run from the repo root, because MODEL_DIR in a profile is a relative path.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
# Pick the profile and layer .env under it. Sourced, never executed; see the
# header of core/profile.sh for why.
source "${ROOT}/core/profile.sh" "${1:-}"

REPO="${MODEL_REPO:-}"
DEST="${MODEL_DIR:-}"
REVISION="${MODEL_REVISION:-main}"
ATTEMPTS="${MODEL_FETCH_ATTEMPTS:-100}"

# Read the file list. `while read` and not `mapfile`: macOS ships bash 3.2,
# which has no mapfile. The `|| [[ -n ... ]]` tail catches a last line that
# carries no newline.
FILES=()
while IFS= read -r name || [[ -n "${name}" ]]; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  FILES+=("${name}")
done < "${PROFILE_DIR}/files.txt"

# Exit before the loop when the list is empty. This is required, not polite:
# bash 3.2 under `set -u` aborts on "${FILES[@]}" when the array has no
# elements.
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Profile ${PROFILE} lists no files."
  echo "The engine loads ${MLX_MODEL} straight from Hugging Face on first serve."
  exit 0
fi

if [[ -z "${REPO}" || -z "${DEST}" ]]; then
  echo "Profile ${PROFILE} lists files but sets no MODEL_REPO or MODEL_DIR." >&2
  echo "Add both to profiles/${PROFILE}/model.env." >&2
  exit 1
fi

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

echo "Fetching ${PROFILE} weights: ${REPO} -> ${DEST}"

for name in "${FILES[@]}"; do
  url="https://huggingface.co/${REPO}/resolve/${REVISION}/${name}"
  out="${DEST}/${name}"
  # files.txt may name a path inside a subfolder, which the mkdir above did
  # not create.
  mkdir -p "$(dirname "${out}")"

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
