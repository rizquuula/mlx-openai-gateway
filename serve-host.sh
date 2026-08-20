#!/usr/bin/env bash
# Thin wrapper. The real script is core/serve.sh. This keeps ./serve-host.sh
# working for shell history, notes, and every ./serve-host.sh in README.md.
#
# exec replaces this process, so signals, the exit status, and
# `pkill -f mlx_vlm.server` behave exactly as they did before the move.
set -euo pipefail
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/core/serve.sh" "$@"
