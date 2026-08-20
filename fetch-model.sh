#!/usr/bin/env bash
# Thin wrapper. The real script is core/fetch.sh. This keeps ./fetch-model.sh
# working for shell history, notes, and every ./fetch-model.sh in README.md.
#
# exec replaces this process, so signals and the exit status behave exactly as
# they did before the move.
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/core/fetch.sh" "$@"
