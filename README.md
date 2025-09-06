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

## Notebooks

- Keep notebooks in `notebooks/`.
- Clear outputs before committing, or use a tool like `nbstripout` to avoid diff noise.

## Security Note

Your team id is a secret. It is stored locally in `icfp_id.json` (which is `.gitignore`d) or read from the `ICFP_TEAM_ID` environment variable. Rotate it if it may have been exposed.
