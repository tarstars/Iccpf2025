# ICFP Contest 2025 — Task Overview

This repository contains solutions and materials for the ICFP Contest 2025.

## Task Summary

The contest task is defined by the official problem statement provided by the organizers. In 2025, the challenge involves solving a computationally intensive, algorithmic problem designed to test creativity and programming skills under time constraints. For full details, refer to the official specification.

**Official task specification:**  
https://icfpcontest2025.github.io/specs/task_from_tex.html

> Please see the above link for the complete problem statement, rules, and example inputs/outputs.

---

Contributions, solutions, and documentation will be added as the contest progresses.

---

## Problem PDF

- Local copy: `docs/ICFPC2025-task.pdf`

The local PDF mirrors the official online specification for convenient offline access.

## Local Setup

- Python tooling is organized under `src/` and `scripts/`.
- The Colab notebook was moved to `notebooks/`.
- Secrets are stored locally and ignored by Git.

Install dependencies:

```
pip install -r requirements.txt
```

## CLI Usage

The helper CLI talks to the contest API and keeps your team id locally.

- Register a team (saves id to `icfp_id.json`):

```
python scripts/icfp_cli.py register --name "Your Team" --pl Python --email you@example.com
```

- Select a problem (uses `ICFP_TEAM_ID` env var or `icfp_id.json`):

```
python scripts/icfp_cli.py select --problem probatio
```

- Explore with one or more plans:

```
python scripts/icfp_cli.py explore --plan "" --plan 0 --plan 1
# or from a file
python scripts/icfp_cli.py explore --file plans.txt
```

You can explicitly pass your team id with `--id`, or set it via environment:

```
export ICFP_TEAM_ID=...  # your secret id
```

- List available problems:

```
PYTHONPATH=. python3 scripts/icfp_cli.py problems
# To see the raw JSON response
PYTHONPATH=. python3 scripts/icfp_cli.py problems --raw
```

## Solving and Submitting

Use the general solver script to select, explore, learn the map, and submit a guess.

- Prereqs:
  - Install deps: `pip install -r requirements.txt`
  - Set your team id via env or `icfp_id.json` (see above)

- Solve a problem (default: `probatio`):

```
PYTHONPATH=. python3 scripts/solve.py --problem probatio
```

- Using a specific Python environment (example):

```
PYTHONPATH=. ~/envs/env312/bin/python3 scripts/solve.py --problem primus
```

The solver will:
- POST `/select` for the chosen problem,
- repeatedly call `/explore` to gather observations,
- reconstruct the map (states, transitions, labels),
- POST `/guess` with the learned map,
- and print whether it was correct.

## Where to Look (Algorithm)

- `scripts/solve.py`
  - Entry point: `main()` parses flags like `--problem`, `--no-submit`, `--max-e-depth`, `--stats`.
  - Core logic: `solve_and_optionally_submit()` implements an L*-style observation table:
    - Closes the set of representative prefixes `S` and refines the suffix set `E` to ensure consistency.
    - Builds transitions and a minimal representative per distinct row.
    - Constructs the map and optionally submits via `/guess`.
  - Key functions to read:
    - `explore_plans()` — batches queries to `/explore` and caches results.
    - `close_table()` and `find_inconsistency()` — maintain closed and consistent observation table.
    - `refine_e_with()` — adds distinguishing suffixes to separate ambiguous states.

- `src/icfp_client.py`
  - Thin HTTP client for the contest API: `select()`, `explore()`, `guess()`, and `list_problems()`.

- `notebooks/Untitled1.ipynb`
  - Contains the exploratory derivation and a working approach (up to aleph) that this CLI closely follows: fixed suffix sets with dynamic expansion, closure on representative rows, targeted probes for missing/mismatched transitions, and mutual-edge map assembly.

- List available problems:

```
PYTHONPATH=. python3 scripts/icfp_cli.py problems
# To see the raw JSON response
PYTHONPATH=. python3 scripts/icfp_cli.py problems --raw
```

## Notebooks

- Keep notebooks in `notebooks/`.
- Clear outputs before committing, or use a tool like `nbstripout` to avoid diff noise.

## Security Note

Your team id is a secret. It is stored locally in `icfp_id.json` (which is `.gitignore`d) or read from the `ICFP_TEAM_ID` environment variable. Rotate it if it may have been exposed.
