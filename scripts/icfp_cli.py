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

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

