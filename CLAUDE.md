# CLAUDE.md

Guidance for Claude Code working in this repo. Read `README.md` for what the
stack does and how a person runs it. This file holds the rules that are easy to
break and expensive to get wrong.

## The stack in one paragraph

Two processes. The MLX engine (`mlx_vlm.server`) runs on the host because macOS
does not pass Metal into containers. The FastAPI gateway runs in Docker and
proxies `/v1/*` to the engine over `host.docker.internal`. Both bind loopback.

## Layout

| Path | Holds |
|---|---|
| `core/` | everything model-agnostic: `serve.sh`, `fetch.sh`, `profile.sh`, `bench.py`, `gateway/` |
| `profiles/<name>/` | one model: `model.env`, `files.txt`, `NOTES.md` |
| `profiles/default` | one line naming the profile used when nobody asks |
| `serve-host.sh`, `fetch-model.sh` | root wrappers that `exec` into `core/` |
| `models/` | weights. Gitignored. Never write here by hand |
| `docker-compose.yml` | stays at the repo root. See the invariants |

Add a model by copying a profile folder. Never add a model by editing `core/`.

## Invariants

Break one of these and something fails quietly or costs 16 GB.

**bash 3.2 only.** macOS `/usr/bin/env bash` is 3.2.57. No `mapfile`, no
`declare -g`, no associative arrays, no `${var,,}`. Test with `/bin/bash`, not
zsh.

**Empty arrays abort under `set -u`.** bash 3.2 treats `"${arr[@]}"` on an empty
array as an unbound variable. Use the `${arr[@]+"${arr[@]}"}` form that
`core/serve.sh` already uses, and guard any new array expansion the same way.

**`core/profile.sh` is sourced at top level. Never wrap it in a function.**
It replays the caller's environment with `eval "$(export -p)"`. `export -p`
emits `declare -x NAME=...`, and bash 3.2 makes `declare` inside a function a
*local* that dies on return. Wrapped in a function, the replay silently loses
the command-line override, so `MLX_THINKING=on ./serve-host.sh` stops working
and nothing reports it.

**`MLX_MODEL` stays relative, and the scripts `cd "${ROOT}"`.** The engine
reports `MLX_MODEL` verbatim at `/v1/models` and matches incoming request
`model` fields against it. Rewriting it to an absolute path changes the id every
client must send, and breaks the README example and `core/bench.py` at once.

**`MODEL_DIR` must byte-match the directory on disk.** A trailing slash, a `./`
prefix, or a case change makes `core/fetch.sh` treat 16 GB of existing weights
as a fresh download target.

**`model.env` is plain `KEY=value`.** Two parsers read it: bash `source` and
`core/bench.py`, which splits on the first `=`. No `export`, no `$expansion`, no
quotes. An empty value is meaningful — `MLX_KV_BITS=` shadows an
`MLX_KV_BITS=8` left in `.env`, because `core/serve.sh` tests knobs with `-n`.

**`docker-compose.yml` stays at the repo root.** Compose derives the project
name from its directory (`mlx`) and discovers `.env` there. Moving it renames
the project and orphans the running network.

**Do not edit `.gitignore` for `profiles/`.** It is already correct: `models/`
is fully ignored and every committed path under `profiles/` is trackable.

**Never touch** `models/`, `.venv/`, `.env`, or `engine.log`. `.env` is the
user's local machine config and is not in git.

## Environment precedence

Highest first: **real environment > `profiles/<name>/model.env` > `.env`**.

The more specific source wins. You type the environment for one run, so it beats
every file. You named the profile this invocation, so it beats the machine-wide
`.env`. This is why a stale `MLX_MODEL` in `.env` cannot override the profile
you asked for.

Profile resolution, first match wins: the argument, then `MLX_PROFILE` in the
environment, then `profiles/default`, then exit 1 listing the profiles.

**`MLX_PROFILE` inside `.env` does nothing.** The scripts resolve the profile
before they read `.env`, so a value there arrives too late. Do not "fix" this by
pre-parsing `.env`; document it instead.

## The engine hot-swaps models at request time

`--model` is only a preload. The engine loads whatever id the request body
names, unloading what it was serving:

```
New text_generation model requested; clearing its existing cache.
Unloading text_generation model: models/Qwen3.8-27B-Uncensored-MLX
Loading model: mlx-community/Qwen3.5-9B-4bit
```

Two consequences:

1. **`/v1/models` does not tell you what is loaded.** It lists every model the
   engine has seen. To learn what is actually serving, grep `engine.log` for
   `Pre-loading language model:`.
2. **A wrong model id is expensive, not cosmetic.** It unloads the served model
   and downloads another. This is why `core/bench.py` reads the id from the
   profile and never asks the engine.

## Conventions

**Comments explain why, not what.** Every file here is deliberately commented
with reasoning. Match that density and voice: short sentences, active voice, one
idea each. A comment that restates the code is worse than none. Read the header
of `core/fetch.sh` to calibrate.

**Commits** use Conventional Commits. Body sentences are imperative, active, and
simple tense. No `Co-Authored-By` and no Claude or Anthropic attribution
trailer. Never push without asking.

**Model knowledge lives in `profiles/`, not in `README.md`.** Numbers, memory
ceilings, and tuning findings belong in that model's `NOTES.md`. The README
links to them.

## Validating without breaking a running stack

The engine and the `mlx-gateway` container are often already running and
serving. Do not restart them to test a change.

| Check | Command |
|---|---|
| Shell syntax | `/bin/bash -n core/serve.sh core/fetch.sh core/profile.sh serve-host.sh fetch-model.sh` |
| Profile resolution | source `core/profile.sh` in a probe script and print `PROFILE` and `MLX_MODEL` |
| End-to-end, safe | `./fetch-model.sh qwen3.8-27b-uncensored` |
| Compose wiring | `docker compose config` |
| Bench id, no HTTP | import `core/bench.py` with `openai` stubbed and call `_active_model()` |

`./fetch-model.sh qwen3.8-27b-uncensored` is the best check available. All 14
files already exist, so it must print `ok` fourteen times and exit 0 having
downloaded zero bytes. **If it starts downloading, stop — the profile path is
wrong.** Confirm with `du -sk models/Qwen3.8-27B-Uncensored-MLX` before and
after.

Do not run `docker compose up` or let `./serve-host.sh` load a model just to
check a banner. Port 8080 is usually taken, and a model load costs minutes and
gigabytes of memory.
