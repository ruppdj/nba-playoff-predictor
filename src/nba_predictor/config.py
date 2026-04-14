"""Load and expose typed configuration from conf/config.yaml."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

# Resolve the project root (two levels up from this file: src/nba_predictor/config.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONF_PATH = PROJECT_ROOT / "conf" / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


class Config:
    """Thin wrapper around the YAML config. Attributes mirror top-level keys."""

    def __init__(self, path: Path = CONF_PATH) -> None:
        self._raw: dict[str, Any] = _load_yaml(path)
        # Top-level keys as attributes (skip keys that are defined as properties)
        _property_names = {name for name, val in type(self).__dict__.items()
                           if isinstance(val, property)}
        for key, value in self._raw.items():
            if key not in _property_names:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config(path={CONF_PATH})"

    # ── Convenience helpers ──────────────────────────────────────────────────

    @property
    def season_range(self) -> list[int]:
        """All seasons in the training range (inclusive)."""
        s = self.seasons
        return list(range(s["start"], s["end"] + 1))

    @property
    def team_name_map(self) -> dict[str, str]:
        return self._raw["team_name_map"]

    def normalize_team(self, name: str) -> str:
        """Return canonical 3-letter abbreviation for any team name variant."""
        mapped = self.team_name_map.get(name)
        if mapped is None:
            raise KeyError(f"Unknown team name '{name}' — add to conf/config.yaml team_name_map")
        return mapped

    def get_era(self, season: int) -> str:
        """Return the era label for a given season year."""
        for era_key, era_cfg in self.eras.items():
            if era_cfg["start"] <= season <= era_cfg["end"]:
                return era_key
        return "unknown"

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def path(self, *keys: str) -> Path:
        """Resolve a nested path from config to an absolute Path."""
        node = self._raw["paths"]
        for k in keys:
            node = node[k]
        return PROJECT_ROOT / node


def get_git_hash() -> str:
    """Return the current git commit hash (for MLflow tagging)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# Module-level singleton — import this everywhere
cfg = Config()
