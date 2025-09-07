#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com"


class ICFPError(RuntimeError):
    pass


@dataclass
class Team:
    id: str

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "Team":
        tid = data.get("id")
        if not isinstance(tid, str) or not tid:
            raise ICFPError("Invalid team id in response")
        return Team(id=tid)


class ICFPClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = base_url or os.getenv("ICFP_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as exc:  # noqa: BLE001
            raise ICFPError(f"Invalid JSON from {url}") from exc

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception as exc:  # noqa: BLE001
            raise ICFPError(f"Invalid JSON from {url}") from exc

    # API methods
    def register(self, name: str, pl: str, email: str) -> Team:
        data = {"name": name, "pl": pl, "email": email}
        resp = self._post("/register", data)
        return Team.from_json(resp)

    def select(self, team_id: str, problem_name: str) -> Dict[str, Any]:
        payload = {"id": team_id, "problemName": problem_name}
        return self._post("/select", payload)

    def explore(self, team_id: str, plans: Iterable[str]) -> Dict[str, Any]:
        payload = {"id": team_id, "plans": list(plans)}
        return self._post("/explore", payload)

    def guess(self, team_id: str, map_obj: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"id": team_id, "map": map_obj}
        return self._post("/guess", payload)

    def list_problems(self) -> List[Any]:
        """Return the list of available problems.

        According to the public notebook, GET /select returns the list.
        """
        data = self._get("/select")
        if not isinstance(data, list):
            raise ICFPError("Unexpected response for problem list: not a list")
        return data


# Local secret helpers
def save_team_id(team_id: str, path: str = "icfp_id.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"id": team_id}, f)


def load_team_id(path: str = "icfp_id.json") -> Optional[str]:
    # Priority: explicit file > env var > None
    if os.path.exists(path):
        try:
            data = json.loads(open(path, "r", encoding="utf-8").read())
            tid = data.get("id")
            if isinstance(tid, str) and tid:
                return tid
        except Exception:
            pass
    env = os.getenv("ICFP_TEAM_ID")
    if env:
        return env
    return None
