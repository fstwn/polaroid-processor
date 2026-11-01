"""Locks on an output folder.

``RunLock``: one batch run (app, CLI or MCP) per output folder. ``file_mutex``: a short lock around
the manifest's read-merge-write, so two writers cannot interleave.

Both are operating-system locks on a byte of an open file (``msvcrt.locking`` / ``fcntl.flock``), so
the system drops them when the process dies: a crashed run never leaves a lock behind, and two
processes cannot both take over one. The lock file also holds the owner's pid, who and when, for the
message a refused run shows.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

LOCK_NAME = "polaroid-run.lock"
_LOCK_AT = 1 << 20     # the locked byte lies beyond the text, so that others can still read the text (Windows)

if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        os.lseek(fd, _LOCK_AT, 0)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
        finally:
            os.lseek(fd, 0, 0)

    def _unlock(fd: int) -> None:
        os.lseek(fd, _LOCK_AT, 0)
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)


class RunLocked(ValueError):
    """Another run is working on the output folder."""


def _open_locked(path: Path, tries: int = 5, delay: float = 0.025) -> Optional[int]:
    """Open ``path`` (created when missing) and take its lock; the descriptor, or None when another
    holder has it for about ``tries * delay`` seconds (a ``holder()`` probe holds it for a moment). The lock
    counts only if the file at ``path`` is still the one opened (the previous holder may have removed it)."""
    for attempt in range(tries + 4):
        fd = os.open(path, os.O_CREAT | os.O_RDWR)
        if not _try_lock(fd):
            os.close(fd)
            if attempt >= tries - 1:
                return None
            time.sleep(delay)
            continue
        try:
            same = os.path.samestat(os.fstat(fd), os.stat(path))
        except OSError:
            same = False
        if same:
            return fd
        _unlock(fd)
        os.close(fd)
    return None


def _read_text(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def holder(out_dir: str) -> Optional[dict]:
    """Who holds the run lock of ``out_dir`` (``{"pid", "who", "started"}``), or None when it is free."""
    path = Path(out_dir) / LOCK_NAME
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        if _try_lock(fd):
            _unlock(fd)
            return None
    finally:
        os.close(fd)
    info = _read_text(path) or {}     # empty while the owner is writing it
    return {"pid": info.get("pid", 0), "who": info.get("who", "unknown"), "started": info.get("started", "")}


class RunLock:
    """``with RunLock(out_dir):`` raises :class:`RunLocked` when another run holds the folder."""

    def __init__(self, out_dir: str, who: Optional[str] = None):
        self.dir = Path(out_dir)
        self.path = self.dir / LOCK_NAME
        self.who = who or os.path.basename(sys.argv[0] or "python")
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = _open_locked(self.path)
        if fd is None:
            info = holder(str(self.dir)) or {"pid": "?", "who": "?", "started": "?"}
            raise RunLocked(f"another run is working on {self.dir} (pid {info['pid']}, {info['who']}, since "
                            f"{info['started']}); wait until it ends. The system drops the lock when that process "
                            f"ends, so a lock without a live run does not exist; if this keeps happening, delete "
                            f"{self.path}")
        try:
            os.ftruncate(fd, 0)
            os.write(fd, json.dumps({"pid": os.getpid(), "who": self.who,
                                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}).encode("utf-8"))
        except BaseException:      # no half-taken lock: nobody could ever release it
            _unlock(fd)
            os.close(fd)
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        _unlock(fd)
        os.close(fd)
        with contextlib.suppress(OSError):     # fails while another process has opened it: that one reuses the file
            self.path.unlink()

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@contextlib.contextmanager
def file_mutex(path: Path, timeout: float = 20.0):
    """Exclusive lock on ``path`` (a file that may be left behind; its content is not used)."""
    path = Path(path)
    end = time.time() + timeout
    while True:
        fd = _open_locked(path)
        if fd is not None:
            break
        if time.time() > end:
            raise TimeoutError(f"{path} is locked by another writer")
        time.sleep(0.02)
    try:
        yield
    finally:
        _unlock(fd)
        os.close(fd)
        with contextlib.suppress(OSError):
            path.unlink()
