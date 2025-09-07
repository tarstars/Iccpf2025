#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from src.icfp_client import ICFPClient, load_team_id
except Exception:  # pragma: no cover
    from icfp_client import ICFPClient, load_team_id  # type: ignore


ALPHA = "012345"


def solve_and_optionally_submit(
    team_id: str,
    problem: str,
    *,
    max_e_depth: int = 2,
    submit: bool = True,
    verbose: bool = True,
    stats: bool = False,
) -> Dict[str, Any]:
    client = ICFPClient()

    # Select problem
    sel = client.select(team_id, problem)
    if verbose:
        print("Selected:", json.dumps(sel, indent=2))

    # Exploration cache: plan -> list[int]
    results: Dict[str, List[int]] = {}
    query_count_seen: Optional[int] = None

    CHUNK = 200

    def explore_plans(plans: Iterable[str]) -> None:
        nonlocal results, query_count_seen
        todo_all = [p for p in plans if p not in results]
        if not todo_all:
            return
        for i in range(0, len(todo_all), CHUNK):
            batch = todo_all[i:i + CHUNK]
            data = client.explore(team_id, batch)
            if not isinstance(data, dict) or "results" not in data or not isinstance(data["results"], list):
                raise RuntimeError(f"Unexpected /explore response: {json.dumps(data)[:200]}")
            arr = data["results"]
            if len(arr) != len(batch):
                raise RuntimeError("/explore result count does not match requested batch size")
            for p, v in zip(batch, arr):
                if isinstance(v, list):
                    # Mask labels to 2 bits as per spec
                    results[p] = [int(x) & 3 for x in v]
            if "queryCount" in data and verbose:
                query_count_seen = data["queryCount"]
                print(f"queryCount: {query_count_seen}")

    def label_at(plan: str, pos: int) -> int:
        row = results.get(plan)
        if row is None:
            raise RuntimeError(f"Missing observation for plan '{plan}'")
        if not (0 <= pos < len(row)):
            raise RuntimeError(f"Observation for plan '{plan}' too short (need index {pos}, got {len(row)})")
        return row[pos]

    # Distinguishing suffix set E with increasing levels
    def make_suffixes(level: int) -> List[str]:
        if level <= 1:
            return [""] + list(ALPHA)
        if level == 2:
            return [""] + list(ALPHA) + [a + b for a in ALPHA for b in ALPHA]
        return [""] + list(ALPHA) + [a + b for a in ALPHA for b in ALPHA] + [a + b + c for a in ALPHA for b in ALPHA for c in ALPHA]

    E_level = 1
    E = make_suffixes(E_level)

    def row_of(prefix: str) -> Tuple[int, ...]:
        return tuple(label_at(p, len(p)) for p in (prefix + s for s in E))

    def ensure_rows_for(prefixes: Iterable[str]) -> None:
        need: List[str] = []
        for p in prefixes:
            for s in E:
                if p + s not in results:
                    need.append(p + s)
        if need:
            explore_plans(need)

    # Seed and prime for current E
    explore_plans([""])
    ensure_rows_for([""])

    def rows_and_reps() -> Tuple[Dict[Tuple[int, ...], List[str]], List[str]]:
        rows: Dict[Tuple[int, ...], List[str]] = {}
        for p in list(results):
            if all(p + s in results for s in E):
                rows.setdefault(row_of(p), []).append(p)
        reps = [min(pl, key=len) for pl in rows.values()]
        reps.sort(key=len)
        return rows, reps

    def build_delta(reps_sorted: List[str]) -> List[List[Optional[int]]]:
        idx_of = {row_of(rep): i for i, rep in enumerate(reps_sorted)}
        n = len(reps_sorted)
        delta: List[List[Optional[int]]] = [[None] * 6 for _ in range(n)]
        for i, rep in enumerate(reps_sorted):
            for a in range(6):
                sp = rep + str(a)
                if all(sp + s in results for s in E):
                    delta[i][a] = idx_of[row_of(sp)]
        return delta

    def partner_issues(delta: List[List[Optional[int]]]) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
        n = len(delta)
        non_mutual: List[Tuple[int, int, int]] = []
        missing: List[Tuple[int, int]] = []
        for i in range(n):
            for a in range(6):
                j = delta[i][a]
                if j is None:
                    missing.append((i, a))
                else:
                    if all(delta[j][b] != i for b in range(6)):
                        non_mutual.append((i, a, j))
        return non_mutual, missing

    def probes_for_gaps(reps_sorted: List[str], delta: List[List[Optional[int]]], non_mutual: List[Tuple[int, int, int]], missing: List[Tuple[int, int]]) -> List[str]:
        need: set[str] = set()
        for (i, a) in missing:
            sp = reps_sorted[i] + str(a)
            for s in E:
                if sp + s not in results:
                    need.add(sp + s)
        for (_i, _a, j) in non_mutual:
            for b in range(6):
                sp = reps_sorted[j] + str(b)
                for s in E:
                    if sp + s not in results:
                        need.add(sp + s)
        return list(need)

    def build_map(reps_sorted: List[str], delta: List[List[Optional[int]]]) -> Dict[str, Any]:
        idx_of = {row_of(rep): i for i, rep in enumerate(reps_sorted)}
        rooms = [row_of(rep)[0] for rep in reps_sorted]
        start_idx = idx_of[row_of("")]
        seen = set()
        connections: List[Dict[str, Dict[str, int]]] = []
        for i in range(len(reps_sorted)):
            for a in range(6):
                j = delta[i][a]
                if j is None:
                    continue
                for b in range(6):
                    if delta[j][b] == i:
                        edge = tuple(sorted([(i, a), (j, b)]))
                        if edge not in seen:
                            seen.add(edge)
                            connections.append({
                                "from": {"room": i, "door": a},
                                "to": {"room": j, "door": b},
                            })
                        break
        return {"rooms": rooms, "startingRoom": start_idx, "connections": connections}

    # Prime reps and their one-step successors
    rows, reps = rows_and_reps()
    ensure_rows_for([r for r in reps] + [r + d for r in reps for d in ALPHA])

    def score(delta: List[List[Optional[int]]]) -> Tuple[int, int, int]:
        n = len(delta)
        filled = sum(1 for row in delta for v in row if v is not None)
        miss = n * 6 - filled
        return (n, filled, miss)

    rounds = 0
    stagnant = 0
    prev: Tuple[int, int, int] = (-1, -1, 1_000_000_000)
    MAX_ROUNDS = 25

    while rounds < MAX_ROUNDS:
        rows, reps = rows_and_reps()
        delta = build_delta(reps)
        non_mutual, missing = partner_issues(delta)
        cur = score(delta)
        if verbose:
            print(f"Round {rounds}: states={cur[0]} filled={cur[1]} missing={cur[2]} non_mutual={len(non_mutual)} E_level={E_level}")

        if not non_mutual and not missing:
            final_map = build_map(reps, delta)
            if verbose:
                print("Learned map:")
                print(json.dumps(final_map, indent=2))
            if stats and query_count_seen is not None:
                print(f"Total queryCount observed: {query_count_seen}")
            if not submit:
                return {"submitted": False, "map": final_map}
            guess_resp = client.guess(team_id, final_map)
            if verbose:
                print("Guess response:", json.dumps(guess_resp, indent=2))
            return guess_resp

        probes = probes_for_gaps(reps, delta, non_mutual, missing)
        if probes:
            explore_plans(probes)
            stagnant = 0
        else:
            stagnant += 1

        made_progress = (cur[0] > prev[0]) or (cur[1] > prev[1]) or (cur[2] < prev[2])
        if not made_progress:
            stagnant += 1
        else:
            stagnant = 0
        prev = cur
        rounds += 1

        if stagnant >= 2 and max(len(x) for x in E) < max_e_depth:
            # Expand E level by one (up to cap)
            E_level = min(max_e_depth, E_level + 1)
            E = make_suffixes(E_level)
            stagnant = 0
            if verbose:
                print(f">>> Expanding suffix set to level {E_level} (|E|={len(E)}) and re-probing reps and exits …")
            need = []
            for rep in reps:
                need += [rep + s for s in E if rep + s not in results]
                for a in ALPHA:
                    sp = rep + a
                    need += [sp + s for s in E if sp + s not in results]
            if need:
                explore_plans(list(set(need)))

    # Fall back: return partial without submitting
    rows, reps = rows_and_reps()
    delta = build_delta(reps)
    partial_map = build_map(reps, delta)
    if verbose:
        print("Stopped without closure; returning partial map")
        print(json.dumps(partial_map, indent=2))
    return {"submitted": False, "map": partial_map}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Solve and (optionally) submit an ICFP 2025 problem")
    p.add_argument("--problem", default="probatio", help="Problem name to solve (default: probatio)")
    p.add_argument("--max-e-depth", type=int, default=2, help="Max distinguishing suffix depth for E (default: 2)")
    p.add_argument("--no-submit", action="store_true", help="Do not submit; only print learned map")
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    p.add_argument("--stats", action="store_true", help="Print query stats if available")
    args = p.parse_args(argv)

    team_id = load_team_id("icfp_id.json") or os.getenv("ICFP_TEAM_ID")
    if not team_id:
        print("Error: team id not found. Provide icfp_id.json or set ICFP_TEAM_ID.", file=sys.stderr)
        return 2

    solve_and_optionally_submit(
        team_id,
        args.problem,
        max_e_depth=args.max_e_depth,
        submit=not args.no_submit,
        verbose=not args.quiet,
        stats=args.stats,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
