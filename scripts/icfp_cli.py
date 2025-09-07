#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

try:
    from src.icfp_client import ICFPClient, load_team_id, save_team_id
except Exception:  # fallback if running as module elsewhere
    # Allow running when installed or packaged differently
    from icfp_client import ICFPClient, load_team_id, save_team_id  # type: ignore


def _resolve_team_id(explicit: Optional[str], id_file: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    path = id_file or "icfp_id.json"
    return load_team_id(path)


def cmd_register(args: argparse.Namespace) -> int:
    client = ICFPClient()
    team = client.register(args.name, args.pl, args.email)
    print(f"REGISTERED. Your SECRET id: {team.id}")
    save_path = args.out or "icfp_id.json"
    save_team_id(team.id, save_path)
    print(f"Saved to {save_path}")
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    team_id = _resolve_team_id(args.id, args.id_file)
    if not team_id:
        print("Error: team id not found. Provide --id, --id-file, or set ICFP_TEAM_ID / icfp_id.json", file=sys.stderr)
        return 2
    client = ICFPClient()
    data = client.select(team_id, args.problem)
    print(json.dumps(data, indent=2))
    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    team_id = _resolve_team_id(args.id, args.id_file)
    if not team_id:
        print("Error: team id not found. Provide --id, --id-file, or set ICFP_TEAM_ID / icfp_id.json", file=sys.stderr)
        return 2
    plans: List[str] = []
    if args.plan:
        plans.extend(args.plan)
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            plans.extend([line.strip() for line in f if line.strip()])
    if not plans:
        print("Error: provide at least one plan via --plan or --file", file=sys.stderr)
        return 2
    client = ICFPClient()
    data = client.explore(team_id, plans)
    print(json.dumps(data, indent=2))
    return 0


def cmd_problems(args: argparse.Namespace) -> int:
    client = ICFPClient()
    items = client.list_problems()
    if args.raw:
        print(json.dumps(items, indent=2))
        return 0
    # Try to render a clean numbered list
    names = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            for key in ("name", "problem", "problemName", "id", "title"):
                if key in item:
                    names.append(str(item[key]))
                    break
            else:
                names.append(json.dumps(item))
        else:
            names.append(str(item))
    print("Available problems:")
    for i, name in enumerate(names):
        print(f"{i:2d}. {name}")
    return 0


def cmd_scores(args: argparse.Namespace) -> int:
    team_id = _resolve_team_id(args.id, args.id_file)
    if not team_id:
        print("Error: team id not found. Provide --id, --id-file, or set ICFP_TEAM_ID / icfp_id.json", file=sys.stderr)
        return 2
    client = ICFPClient()
    data = client.scores(team_id)
    if args.raw:
        print(json.dumps(data, indent=2))
        return 0
    print("Your best expeditions per problem (lower is better):")
    for k in sorted(data.keys()):
        print(f"- {k}: {data[k]}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    problem = args.problem or "global"
    client = ICFPClient()
    rows = client.leaderboard(problem)
    if args.raw:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"Leaderboard for {problem}:")
    # Sort: for global, higher score is better; for problems, lower expeditions is better
    keyf = (lambda r: (-int(r.get("score", 0)) if problem == "global" else int(r.get("score", 10**9))))
    rows_sorted = sorted(rows, key=keyf)
    print("Rank  Score  Team (PL)")
    rank = 1
    prev = None
    display = 1
    for r in rows_sorted:
        if r.get("score") is None:
            continue
        sc = r.get("score")
        if prev is not None and sc != prev:
            display = rank
        print(f"{display:>4}  {sc:>5}  {r.get('teamName','<no name>')} ({r.get('teamPl','<no pl>')})")
        prev = sc
        rank += 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="icfp", description="ICFP 2025 API helper CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # register
    pr = sub.add_parser("register", help="Register a team")
    pr.add_argument("--name", required=True)
    pr.add_argument("--pl", required=True, help="Programming language")
    pr.add_argument("--email", required=True)
    pr.add_argument("--out", help="Where to save the team id (default: icfp_id.json)")
    pr.set_defaults(func=cmd_register)

    # select
    ps = sub.add_parser("select", help="Select a problem")
    ps.add_argument("--problem", required=True, help="Problem name (e.g., probatio)")
    ps.add_argument("--id", help="Team id (overrides env/file)")
    ps.add_argument("--id-file", help="Path to file containing {\"id\": \"...\"}")
    ps.set_defaults(func=cmd_select)

    # explore
    pe = sub.add_parser("explore", help="Explore with one or more plans")
    pe.add_argument("--plan", action="append", help="Plan string; can be repeated")
    pe.add_argument("--file", help="File with one plan per line")
    pe.add_argument("--id", help="Team id (overrides env/file)")
    pe.add_argument("--id-file", help="Path to file containing {\"id\": \"...\"}")
    pe.set_defaults(func=cmd_explore)

    # problems
    pp = sub.add_parser("problems", help="List available problems")
    pp.add_argument("--raw", action="store_true", help="Print raw JSON response")
    pp.set_defaults(func=cmd_problems)

    # scores
    psc = sub.add_parser("scores", help="Show your current best scores per problem (expeditions)")
    psc.add_argument("--id", help="Team id (overrides env/file)")
    psc.add_argument("--id-file", help="Path to file containing {\"id\": \"...\"}")
    psc.add_argument("--raw", action="store_true", help="Print raw JSON response")
    psc.set_defaults(func=cmd_scores)

    # leaderboard
    plb = sub.add_parser("leaderboard", help="Show leaderboard for a problem or 'global'")
    plb.add_argument("--problem", help="Problem name (or 'global')")
    plb.add_argument("--raw", action="store_true", help="Print raw JSON response")
    plb.set_defaults(func=cmd_leaderboard)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
