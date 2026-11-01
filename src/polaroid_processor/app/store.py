"""Per-user app data: recent projects (input and output folder) and recent input folders."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

from ..manifest import now

MAX_PROJECTS = 40
MAX_INPUTS = 12


def default_config_path() -> Path:
    """``%APPDATA%/polaroid-processor/app.json``; ``~/.polaroid-processor/app.json`` without APPDATA."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) / "polaroid-processor" if appdata else Path.home() / ".polaroid-processor"
    return base / "app.json"


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


class Store:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_config_path()
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
        return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---- projects --------------------------------------------------------
    def projects(self) -> List[dict]:
        """Recent projects, newest first: ``{"input": str or None, "output": str, "opened_at": str}``."""
        with self._lock:
            return list(self._load().get("projects", []))

    def touch_project(self, input_dir: Optional[str], output: str) -> None:
        with self._lock:
            data = self._load()
            old = [p for p in data.get("projects", []) if same_path(p["output"], output)]
            rest = [p for p in data.get("projects", []) if not same_path(p["output"], output)]
            entry = {"input": input_dir, "output": output, "opened_at": now()}
            if old and old[0].get("settings"):
                entry["settings"] = old[0]["settings"]       # kept until the project has a manifest of its own
            data["projects"] = ([entry] + rest)[:MAX_PROJECTS]
            if input_dir:
                ins = [p for p in data.get("recent_inputs", []) if not same_path(p, input_dir)]
                data["recent_inputs"] = ([input_dir] + ins)[:MAX_INPUTS]
            self._save(data)

    def forget_project(self, output: str) -> None:
        with self._lock:
            data = self._load()
            data["projects"] = [p for p in data.get("projects", []) if not same_path(p["output"], output)]
            self._save(data)

    # ---- settings ----------------------------------------------------------
    def settings_defaults(self) -> dict:
        """The panel fields the user chose last: the start for new projects."""
        with self._lock:
            return dict(self._load().get("settings_defaults", {}))

    def set_settings_defaults(self, panel: dict) -> None:
        with self._lock:
            data = self._load()
            data["settings_defaults"] = panel
            self._save(data)

    def project_settings(self, output: str) -> Optional[dict]:
        """Settings of a project that has no manifest yet (a manifest holds them afterwards)."""
        with self._lock:
            for p in self._load().get("projects", []):
                if same_path(p["output"], output):
                    return p.get("settings")
        return None

    def set_project_settings(self, output: str, settings: dict) -> None:
        with self._lock:
            data = self._load()
            for p in data.get("projects", []):
                if same_path(p["output"], output):
                    p["settings"] = settings
            self._save(data)

    def recent_inputs(self) -> List[str]:
        with self._lock:
            return list(self._load().get("recent_inputs", []))
