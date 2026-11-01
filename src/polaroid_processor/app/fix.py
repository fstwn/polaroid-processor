"""The drag-to-fix view inside the app: the review page's backend, tied to the open project."""
from __future__ import annotations

import logging
from pathlib import Path

from ..manifest import Manifest
from ..review.server import ReviewApp
from .project import Project

log = logging.getLogger("polaroid_processor.app")


class FixApp(ReviewApp):
    """:class:`ReviewApp` for every scan of a project; a saved or accepted outline is judged again and
    its crop preview renewed before the record goes back to the page."""

    api_prefix = "/api/fix/scans"

    def __init__(self, project: Project, author: str = "human"):
        super().__init__(project.output, show_all=True, author=author)
        self.project = project

    def _saved(self, key: str, rec: dict) -> dict:
        try:
            e = Manifest(self.output).get(key) or {}
            ident, rgb = self._cache
            same = ident is not None and ident[0] == str(Path(e.get("input", "")))
            self.project.after_fix(key, rgb if same else None)
        except Exception:  # noqa: BLE001  the fix is stored; the preview is renewed in the background
            log.exception("after fix of %s", key)
        return super()._saved(key, rec)
