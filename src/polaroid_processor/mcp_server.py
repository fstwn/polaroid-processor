"""MCP server over stdio: the library's commands as tools for agents.

Dependency-free implementation of the Model Context Protocol (JSON-RPC 2.0, one message
per line on stdin/stdout). Start with ``polaroid-mcp``; register it in Claude Code with
``claude mcp add polaroid -- polaroid-mcp``.

Tools: ``polaroid_process``, ``polaroid_detect``, ``polaroid_inspect`` (returns the review
sheet as an image), ``polaroid_fix``, ``polaroid_status``. Inputs are never written to.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
import traceback
from typing import Any, BinaryIO, Callable, Dict, List, Optional

import cv2
import numpy as np

from . import __version__, api
from .pipeline import collect_inputs

log = logging.getLogger("polaroid_processor.mcp")

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = """\
Crops and rectifies flatbed scans of instant prints (Polaroid 600/i-Type/SX-70, Go) lying in a
black holder. Workflow: polaroid_process the input folder into an output folder (one output
folder per input folder); for every scan with status needs_review call polaroid_inspect,
compare the green line with the print edge, then polaroid_fix (set/nudge/shift corners, or
accept) with a note saying why. Coordinates are full-resolution scan pixels; shift is positive
outward. Warnings starting with hint: name a concrete fix. Escalate to a human when the edge is
not visible even in stretched tiles or is physically curved. Never write into input folders."""

_STR_LIST = {"type": "array", "items": {"type": "string"}}
_POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
_CORNER_MAP = {"type": "object", "properties": {c: _POINT for c in ("tl", "tr", "br", "bl")},
               "additionalProperties": False}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "polaroid_process",
        "title": "Process scans",
        "description": (
            "Detect, rectify and export scans into an output folder (TIFF master plus JPG/PNG derivatives, "
            "manifest polaroid-manifest.json). Resumable: unchanged scans are skipped. Returns a summary and "
            "per-scan results; status needs_review means check the scan with polaroid_inspect. Can take "
            "several seconds per scan."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {**_STR_LIST, "description": "scan files or folders (.tif/.tiff)"},
                "output": {"type": "string", "description": "output folder; holds the manifest"},
                "recursive": {"type": "boolean", "default": False},
                "margin_mm": {"type": "number", "default": 0.2, "description": "crop this far inside the edge"},
                "derivatives": {**_STR_LIST, "description": "presets NAME=SIZE[:FMT[:Q]], e.g. web=2048:jpg:88"},
                "no_derivatives": {"type": "boolean", "default": False},
                "ppi": {"type": "number", "description": "true scan resolution"},
                "profile": {"type": "string", "description": "scanner profile name"},
                "overlay": {"type": "boolean", "default": False},
                "strict": {"type": "boolean", "default": False, "description": "do not export flagged scans"},
                "force": {"type": "boolean", "default": False},
                "redetect": {"type": "boolean", "default": False},
                "dry_run": {"type": "boolean", "default": False, "description": "detect and evaluate only"},
                "flatten": {"type": "string", "enum": ["auto", "off"], "default": "auto",
                            "description": "unwarp curled prints along their traced edges"},
                "shadows": {"type": "string", "enum": ["auto", "off"], "default": "auto",
                            "description": "remove edge shadows on plain frames"},
                "workers": {"type": "integer", "minimum": 1},
                "details": {"type": "string", "enum": ["flagged", "all", "none"], "default": "flagged",
                            "description": "full records for flagged scans only (default), all scans, or none"},
            },
            "required": ["inputs"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
    {
        "name": "polaroid_detect",
        "title": "Detect outlines",
        "description": "Detect print outlines in scans without writing anything.",
        "inputSchema": {
            "type": "object",
            "properties": {"inputs": _STR_LIST, "recursive": {"type": "boolean", "default": False}},
            "required": ["inputs"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "polaroid_inspect",
        "title": "Inspect a scan",
        "description": (
            "Review sheet of one scan as an image: eight full-resolution tiles (corners and edge midpoints) "
            "with a grid in absolute scan pixels, the current crop line in green, plus an overview. Dark "
            "tiles are contrast-stretched. Use it to verify a flagged scan and to read exact coordinates "
            "for polaroid_fix."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scan": {"type": "string", "description": "file name or stem (with output), or a scan path"},
                "output": {"type": "string", "description": "output folder whose manifest holds the scan"},
                "window": {"type": "integer", "default": 360, "description": "tile size in scan pixels"},
                "zoom": {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
                "grid": {"type": "integer", "default": 50},
                "stretch": {"type": "string", "enum": ["auto", "on", "off"], "default": "auto"},
                "save": {"type": "boolean", "default": True,
                         "description": "also save the PNG to <output>/_inspect/ (only with output)"},
            },
            "required": ["scan"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "polaroid_fix",
        "title": "Fix a scan outline",
        "description": (
            "Correct or accept the outline of one scan in the manifest and (by default) re-export it. "
            "set: absolute corner positions; nudge: corner offsets; shift: move a whole side outward "
            "(positive) or inward (negative); accept: mark the current outline as reviewed; reset: back "
            "to automatic detection. Every change is logged with author, time, note and previous value."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "output": {"type": "string"},
                "scan": {"type": "string"},
                "set": {**_CORNER_MAP, "description": "e.g. {\"bl\": [265, 13868]}"},
                "nudge": {**_CORNER_MAP, "description": "e.g. {\"tl\": [4, -2]}"},
                "shift": {"type": "object",
                          "properties": {s: {"type": "number"} for s in api.SIDES},
                          "additionalProperties": False, "description": "e.g. {\"bottom\": -59}"},
                "accept": {"type": "boolean", "default": False},
                "reset": {"type": "boolean", "default": False},
                "author": {"type": "string", "default": "agent"},
                "note": {"type": "string", "description": "why; shown in the history"},
                "export": {"type": "boolean", "default": True},
            },
            "required": ["output", "scan"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "polaroid_status",
        "title": "Batch status",
        "description": "State of an output folder from its manifest: counts and per-scan status.",
        "inputSchema": {
            "type": "object",
            "properties": {"output": {"type": "string"},
                           "needs_review": {"type": "boolean", "default": True,
                                            "description": "only list scans that need review"}},
            "required": ["output"],
        },
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
]

_BRIEF = ("file", "status", "warnings", "error", "quad", "quad_source", "size_px", "scale", "format", "flatten",
          "shadows", "outputs",
          "exported")


def _brief(rec: dict) -> dict:
    return {k: rec[k] for k in _BRIEF if k in rec}


def _text(obj: Any) -> dict:
    return {"type": "text", "text": json.dumps(obj, ensure_ascii=False, indent=1)}


def _jpeg(rgb: np.ndarray, quality: int = 88) -> dict:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the review image")
    return {"type": "image", "data": base64.b64encode(buf.tobytes()).decode("ascii"), "mimeType": "image/jpeg"}


# --------------------------------------------------------------------------- tool implementations
def tool_process(args: dict, progress: Callable[[int, int, str], None]) -> dict:
    inputs = args["inputs"]
    settings = api.make_settings(
        margin_mm=args.get("margin_mm", 0.2), derivatives=args.get("derivatives"),
        no_derivatives=args.get("no_derivatives", False), overlay=args.get("overlay", False),
        ppi=args.get("ppi"), profile=args.get("profile"), strict=args.get("strict", False),
        force=args.get("force", False), redetect=args.get("redetect", False), dry_run=args.get("dry_run", False),
        flatten=args.get("flatten", "auto"), shadows=args.get("shadows", "auto"),
        workers=args.get("workers"))
    recursive = args.get("recursive", False)
    total = len(collect_inputs(list(inputs), recursive))
    results: List[dict] = []
    done = [0]

    def job_done(key: str, what: str) -> None:
        done[0] += 1
        progress(done[0], total, f"{key}: {what}")

    summary = api.process(inputs, args.get("output"), settings, recursive=recursive, emit=results.append,
                          progress=job_done)
    details = args.get("details", "flagged")
    if details == "all":
        listed = [_brief(r) for r in results]
    elif details == "flagged":
        listed = [_brief(r) if r.get("status") != "ok" else {"file": r["file"], "status": "ok"} for r in results]
    else:
        listed = []
    out = {"summary": summary, "results": listed}
    return {"content": [_text(out)], "structuredContent": out, "isError": bool(summary.get("failed"))}


def tool_detect(args: dict, progress) -> dict:
    recs = api.detect_scans(args["inputs"], recursive=args.get("recursive", False))
    out = {"detections": recs}
    return {"content": [_text(out)], "structuredContent": out,
            "isError": any(r["status"] == "failed" for r in recs)}


def tool_inspect(args: dict, progress) -> dict:
    output = args.get("output")
    rec, sheet = api.inspect_scan(args["scan"], output, write=bool(output) and args.get("save", True),
                                  window=int(args.get("window", 360)), zoom=int(args.get("zoom", 1)),
                                  grid=int(args.get("grid", 50)), stretch=args.get("stretch", "auto"))
    return {"content": [_text(rec), _jpeg(sheet)], "structuredContent": rec, "isError": False}


def tool_fix(args: dict, progress) -> dict:
    rec = api.fix_scan(args["output"], args["scan"], sets=args.get("set"), nudges=args.get("nudge"),
                       shifts=args.get("shift"), accept=args.get("accept", False), reset=args.get("reset", False),
                       author=args.get("author", "agent"), note=args.get("note", ""),
                       export=args.get("export", True))
    failed = bool((rec.get("summary") or {}).get("failed"))
    return {"content": [_text(rec)], "structuredContent": rec, "isError": failed}


def tool_status(args: dict, progress) -> dict:
    recs: List[dict] = []
    summary = api.status(args["output"], needs_review=args.get("needs_review", True), emit=recs.append)
    out = {"summary": summary, "scans": [_brief(r) for r in recs]}
    return {"content": [_text(out)], "structuredContent": out, "isError": False}


HANDLERS = {"polaroid_process": tool_process, "polaroid_detect": tool_detect, "polaroid_inspect": tool_inspect,
            "polaroid_fix": tool_fix, "polaroid_status": tool_status}


# --------------------------------------------------------------------------- protocol
class Server:
    def __init__(self, out: BinaryIO):
        self.out = out

    def send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n")
        self.out.flush()

    def reply(self, mid: Any, result: Optional[dict] = None, error: Optional[dict] = None) -> None:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": mid}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result if result is not None else {}
        self.send(msg)

    def handle(self, msg: dict) -> None:
        method = msg.get("method")
        mid = msg.get("id")
        params = msg.get("params") or {}
        if method is None:  # a response to something we never sent
            return
        if mid is None:  # notification: initialized, cancelled, ...
            return
        try:
            if method == "initialize":
                asked = params.get("protocolVersion")
                version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
                self.reply(mid, {"protocolVersion": version,
                                 "capabilities": {"tools": {"listChanged": False}},
                                 "serverInfo": {"name": "polaroid-processor", "version": __version__},
                                 "instructions": INSTRUCTIONS})
            elif method == "ping":
                self.reply(mid, {})
            elif method == "tools/list":
                self.reply(mid, {"tools": TOOLS})
            elif method == "tools/call":
                if params.get("name") not in HANDLERS:
                    self.reply(mid, error={"code": -32602, "message": f"unknown tool: {params.get('name')}"})
                else:
                    self.reply(mid, self.call(params))
            else:
                self.reply(mid, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as e:  # noqa: BLE001  protocol-level failure
            log.error("request %s failed: %s", method, traceback.format_exc())
            self.reply(mid, error={"code": -32603, "message": f"{type(e).__name__}: {e}"})

    def call(self, params: dict) -> dict:
        name = params["name"]
        handler = HANDLERS[name]
        args = params.get("arguments") or {}
        token = (params.get("_meta") or {}).get("progressToken")

        def progress(done: int, total: int, message: str) -> None:
            if token is not None:
                self.send({"jsonrpc": "2.0", "method": "notifications/progress",
                           "params": {"progressToken": token, "progress": done, "total": total,
                                      "message": message}})

        try:
            return handler(args, progress)
        except (api.UsageError, ValueError, KeyError, FileNotFoundError) as e:
            msg = f"missing argument {e}" if isinstance(e, KeyError) else str(e)
            return {"content": [{"type": "text", "text": f"error: {msg}"}], "isError": True}
        except Exception as e:  # noqa: BLE001  tool failure is reported to the agent, not as a protocol error
            log.error("tool %s failed: %s", name, traceback.format_exc())
            return {"content": [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}], "isError": True}

    def serve(self, inp: BinaryIO) -> None:
        for raw in inp:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                self.reply(None, error={"code": -32700, "message": f"parse error: {e}"})
                continue
            for m in (msg if isinstance(msg, list) else [msg]):
                if isinstance(m, dict):
                    self.handle(m)


def main() -> int:
    # stdout belongs to the protocol: anything else printed goes to stderr
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    Server(out).serve(sys.stdin.buffer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
