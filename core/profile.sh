#!/usr/bin/env bash
# Resolve one model profile, then layer its environment.
#
# Source this file. Never execute it. Every statement stays at the top level
# on purpose: `export -p` emits `declare -x NAME=...`, and bash 3.2 makes
# `declare` inside a function a local that dies on return. Wrapped in a
# function, the replay below silently loses the command-line override.
#
# The caller sets ROOT to the repo root before it sources this file.
# On return, PROFILE names the profile and PROFILE_DIR points at its folder.

# Profile selection, first match wins. The argument is what you type for one
# run, so it beats MLX_PROFILE, which you exported for the whole shell.
PROFILE="${1:-}"
if [[ -z "${PROFILE}" ]]; then
  PROFILE="${MLX_PROFILE:-}"
fi
# profiles/default names the model you get when you ask for nothing.
if [[ -z "${PROFILE}" && -f "${ROOT}/profiles/default" ]]; then
  PROFILE="$(tr -d '[:space:]' < "${ROOT}/profiles/default")"
fi

_profile_resolved="${PROFILE}"
PROFILE_DIR="${ROOT}/profiles/${PROFILE}"
# model.env is the file that makes a folder a profile. Test for it, not for the
# folder, so a half-made directory fails loudly instead of serving nothing.
if [[ -z "${PROFILE}" || ! -f "${PROFILE_DIR}/model.env" ]]; then
  echo "Unknown profile '${PROFILE}'. Available:" >&2
  for _profile_candidate in "${ROOT}"/profiles/*/; do
    [[ -f "${_profile_candidate}model.env" ]] || continue
    echo "  $(basename "${_profile_candidate}")" >&2
  done
  exit 1
fi

# Precedence, highest first: real environment, profile model.env, root .env.
#
# The real environment must beat every file, because it is what you type for
# one run: `MLX_KV_BITS=8 ./serve-host.sh` has to work without editing a file
# that the next run still reads.
#
# The profile beats .env because you selected it by name this invocation.
# Today .env carries MLX_MODEL=models/Qwen3.8-27B-Uncensored-MLX. If .env won,
# `./serve-host.sh qwen3.5-9b` would load the 27B and say nothing.
#
# So the files load in reverse precedence and the snapshot replays last.
# `set -a` exports what the files define, which is how the engine and
# core/bench.py see the values.
_env_snapshot="$(export -p)"
set -a
# Keep the long form. `[[ -f x ]] && source x` returns 1 when the file is
# missing, and as a final statement that trips set -e.
if [[ -f "${ROOT}/.env" ]]; then
  source "${ROOT}/.env"
fi
source "${PROFILE_DIR}/model.env"
set +a
eval "${_env_snapshot}"
unset _env_snapshot

# The replay restores every exported name, so it would overwrite PROFILE if the
# caller happened to export that name. Set both back from the resolved value.
PROFILE="${_profile_resolved}"
PROFILE_DIR="${ROOT}/profiles/${PROFILE}"
unset _profile_resolved
