"""Batch processing: detect, export, resolve scale, review. Resumable via the manifest."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .detection import DetectionError, DetectParams, detect, quad_size
from .formats import match_format
from .geometry import rectify
from .imageio import (SCAN_SUFFIXES, file_fingerprint, load_rgb, read_meta, resize_long_edge, set_resolution,
                      write_derivative, write_tiff)
from .manifest import Manifest, now, scan_key
from .render import overlay
from .scale import Scale, batch_median_scale, explicit_scale, scale_from_size, scale_from_tag

log = logging.getLogger("polaroid_processor")

GB = 1024 ** 3


# --------------------------------------------------------------------------- settings
@dataclass(frozen=True)
class Derivative:
    name: str
    long_edge: Optional[int] = None
    scale: Optional[float] = None
    fmt: str = "jpg"
    quality: int = 92

    @staticmethod
    def parse(spec: str) -> "Derivative":
        """``name=SIZE[:FORMAT[:QUALITY]]``; SIZE is a long edge in px or ``x0.5`` for a scale."""
        if "=" not in spec:
            raise ValueError(f"derivative {spec!r}: expected name=SIZE[:FORMAT[:QUALITY]]")
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        size = parts[0].strip().lower()
        long_edge, scale = None, None
        if size.startswith("x"):
            scale = float(size[1:])
        elif size not in ("full", "1", "x1"):
            long_edge = int(size)
        fmt = parts[1].strip().lower() if len(parts) > 1 and parts[1] else "jpg"
        if fmt == "jpeg":
            fmt = "jpg"
        if fmt not in ("jpg", "png"):
            raise ValueError(f"derivative {spec!r}: format must be jpg or png")
        quality = int(parts[2]) if len(parts) > 2 and parts[2] else 92
        return Derivative(name.strip(), long_edge, scale, fmt, quality)


DEFAULT_DERIVATIVES = (Derivative("high", 6000, None, "jpg", 95), Derivative("web", 2048, None, "jpg", 88))


@dataclass(frozen=True)
class ReviewRules:
    edge_mad_mm: float = 0.3
    edge_bow_mm: float = 0.4
    min_inlier_share: float = 0.6
    size_tolerance: float = 0.015  # against the nominal print format
    parallel_deg: float = 0.3  # opposite sides of the outline
    size_outlier_mm: float = 0.35  # against the median print of the same scan size (>= 5 scans)


@dataclass
class Settings:
    margin_mm: float = 0.2
    derivatives: Tuple[Derivative, ...] = DEFAULT_DERIVATIVES
    interpolation: str = "cubic"
    tiff_compression: str = "zlib"
    overlay: bool = False
    ppi: Optional[float] = None
    profile: Optional[str] = None
    profiles_file: Optional[str] = None
    strict: bool = False
    force: bool = False
    redetect: bool = False
    dry_run: bool = False
    workers: Optional[int] = None
    rules: ReviewRules = field(default_factory=ReviewRules)
    detect_params: DetectParams = field(default_factory=DetectParams)

    def export_key(self, quad: np.ndarray, scale: Scale) -> str:
        payload = {
            "v": __version__,
            "margin_mm": self.margin_mm,
            "derivatives": [asdict(d) for d in self.derivatives],
            "interpolation": self.interpolation,
            "tiff_compression": self.tiff_compression,
            "overlay": self.overlay,
            "quad": np.round(quad, 2).tolist(),
            "px_per_mm": round(scale.px_per_mm, 4),
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- helpers
def collect_inputs(paths: Sequence[str], recursive: bool = False) -> List[Path]:
    out: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            out.extend(sorted(q for q in it if q.is_file() and q.suffix.lower() in SCAN_SUFFIXES))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(str(p))
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def available_memory() -> Optional[int]:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return int(st.ullAvailPhys)
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        return None
    return None


def auto_workers(n_jobs: int, per_worker_bytes: int, requested: Optional[int] = None) -> int:
    if requested:
        return max(1, min(requested, n_jobs))
    cpu = max(1, (os.cpu_count() or 2) - 1)
    mem = available_memory()
    by_mem = max(1, int(mem // per_worker_bytes)) if mem else 2
    return max(1, min(cpu, by_mem, n_jobs))


def bytes_per_worker(width: int, height: int, bits: int = 8) -> int:
    """Rough peak: decoded scan, warped copy and encoder buffers."""
    return int(width * height * 3 * (bits // 8) * 3.2) + 300 * 1024 ** 2


# --------------------------------------------------------------------------- workers
def _export(rgb: np.ndarray, quad: np.ndarray, px_per_mm: float, icc: Optional[bytes], key: str, ex: dict) -> dict:
    """Rectify and write master + derivatives (+ overlay). Returns outputs and their px/mm."""
    img = rectify(rgb, quad, ex["margin_mm"] * px_per_mm, ex["interpolation"])
    stem = Path(key).stem
    out_dir = Path(ex["out_dir"])
    outputs, ppm = {}, {}
    tiff_path = out_dir / f"{stem}.tif"
    write_tiff(str(tiff_path), img, px_per_mm, icc, ex["tiff_compression"])
    outputs["tiff"], ppm["tiff"] = str(tiff_path), px_per_mm
    for d in ex["derivatives"]:
        small = resize_long_edge(img, d["long_edge"], d["scale"])
        f = small.shape[1] / img.shape[1]
        p = out_dir / d["name"] / f"{stem}.{d['fmt']}"
        write_derivative(str(p), small, d["fmt"], d["quality"], px_per_mm * f * 25.4, icc)
        outputs[d["name"]], ppm[d["name"]] = str(p), px_per_mm * f
    if ex["overlay"]:
        p = out_dir / "_overlays" / f"{stem}.jpg"
        write_derivative(str(p), overlay(rgb, quad), "jpg", 85, None)
        outputs["overlay"] = str(p)
    return {"outputs": outputs, "output_ppm": ppm, "output_size_px": [int(img.shape[1]), int(img.shape[0])]}


def _job(job: dict) -> dict:
    """One decode per scan: detect if needed, export if requested.

    ``job["scale"]`` is the provisional px/mm (explicit or from earlier runs); when None the
    worker uses a plausible TIFF tag or the scan's own size. The batch fixes it afterwards.
    """
    t0 = time.perf_counter()
    path = job["path"]
    res = {"key": job["key"], "path": path}
    try:
        meta = read_meta(path)
        st = os.stat(path)
        res.update(meta={"width": meta.width, "height": meta.height, "bits": meta.bits,
                         "dpi_tag": list(meta.dpi_tag) if meta.dpi_tag else None,
                         "has_icc": meta.icc_profile is not None},
                   size_bytes=st.st_size, mtime=st.st_mtime)
        if job["detect"]:
            res["fingerprint"] = file_fingerprint(path)
        rgb = load_rgb(path)
        det = job.get("detection")
        if job["detect"]:
            try:
                det = detect(rgb, job["detect_params"]).to_dict()
            except DetectionError as e:
                res["error"] = f"detection failed: {e}"
                return res
            res["detection"] = det
        quad = np.array(job["override"] if job.get("override") else det["quad"], dtype=np.float64)
        if job["export"]:
            if job.get("scale"):
                px = job["scale"]
            else:
                s = scale_from_tag(meta.dpi_tag, quad_size(quad)) or scale_from_size(quad_size(quad))
                px = s.px_per_mm
            res.update(_export(rgb, quad, px, meta.icc_profile, job["key"], job["export_settings"]))
            res["export_px_per_mm"] = px
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        res["time_s"] = round(time.perf_counter() - t0, 2)
    return res


def _run_pool(fn: Callable, jobs: List[dict], workers: int):
    if workers <= 1 or len(jobs) <= 1:
        for j in jobs:
            yield fn(j)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for fut in as_completed(futs):
            yield fut.result()


# --------------------------------------------------------------------------- evaluation
def _group(e: dict) -> Tuple[int, int]:
    """Scans of the same pixel size come from the same scanner setting."""
    m = e.get("meta") or {}
    return m.get("width", 0), m.get("height", 0)


def _sizes_by_group(manifest: Manifest) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    sizes: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for _, e in manifest.items():
        q = Manifest.effective_quad(e)
        if q is not None:
            sizes.setdefault(_group(e), []).append(quad_size(q))
    return sizes


def reference_sizes(manifest: Manifest, min_scans: int = 5) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Median print size in px per scan-size group, for groups with at least ``min_scans`` scans."""
    return {g: tuple(float(v) for v in np.median(np.array(s), axis=0))
            for g, s in _sizes_by_group(manifest).items() if len(s) >= min_scans}


def _side_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _inner_hints(entry: dict, sides: Tuple[str, str], excess_px: float, tol_px: float) -> List[str]:
    """Suggest moving a side onto its inner line when that explains an oversized print."""
    edges = (entry.get("detection") or {}).get("edges", {})
    hints = []
    for side in sides:
        depth = (edges.get(side) or {}).get("inner_px")
        if depth and abs(depth - excess_px) <= tol_px:
            hints.append(f"hint:fix --shift {side}=-{depth:.0f} (holder rim parallel to the print)")
    return hints


def resolve_scales(manifest: Manifest, keys: Sequence[str], settings: Settings) -> Dict[str, Scale]:
    """Scale per scan: explicit > plausible TIFF tag > median of all scans with the same scan size."""
    explicit = explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    scales: Dict[str, Scale] = {}
    pending: List[str] = []
    for k in keys:
        e = manifest.get(k) or {}
        quad = Manifest.effective_quad(e)
        if quad is None:
            continue
        if explicit:
            scales[k] = explicit
            continue
        tag = (e.get("meta") or {}).get("dpi_tag")
        s = scale_from_tag(tuple(tag) if tag else None, quad_size(quad))
        if s:
            scales[k] = s
        else:
            pending.append(k)

    sizes_by_group = _sizes_by_group(manifest)
    for k in pending:
        e = manifest.get(k)
        sizes = sizes_by_group.get(_group(e), [])
        if len(sizes) >= 3:
            scales[k] = batch_median_scale(sizes)
        else:
            scales[k] = scale_from_size(quad_size(Manifest.effective_quad(e)))
    return scales


def evaluate(entry: dict, scale: Scale, rules: ReviewRules,
             ref_size: Optional[Tuple[float, float]] = None,
             margin_mm: float = 0.2) -> Tuple[str, List[str], Optional[dict]]:
    """Status and warnings for one scan. ``ref_size`` is the median print size of its scan-size group;
    ``margin_mm`` is the crop margin, which hides edge points that lie at most that far inside."""
    warnings: List[str] = []
    quad = Manifest.effective_quad(entry)
    fm = match_format(quad_size(quad), scale.px_per_mm)
    fmt = None
    if fm:
        fmt = {"name": fm.format.name, "size_mm": list(fm.size_mm), "error": fm.error}
    review = False
    override = entry.get("override")
    if scale.source in ("batch-median", "single-scan"):
        warnings.append(f"scale_inferred:{scale.source}")
    if scale.source == "single-scan":
        warnings.append("size_unverified")
    elif fm and fm.error > rules.size_tolerance:
        warnings.append(f"size_mismatch:{fm.error * 100:.1f}%")
        review = not override
    q = quad
    for name, (a0, a1, b0, b1) in (("top-bottom", (q[0], q[1], q[3], q[2])),
                                   ("left-right", (q[0], q[3], q[1], q[2]))):
        diff = abs(_side_angle_deg(a0, a1) - _side_angle_deg(b0, b1))
        if diff > rules.parallel_deg:
            warnings.append(f"not_parallel:{name}:{diff:.2f}deg")
            review = review or not override
    size = quad_size(quad)
    if ref_size and all(abs(v / r - 1) < 0.03 for v, r in zip(size, ref_size)):  # same format and orientation
        for axis, v, r in zip(("width", "height"), size, ref_size):
            dev = (v - r) / scale.px_per_mm
            if abs(dev) > rules.size_outlier_mm:
                warnings.append(f"size_outlier:{axis}:{dev:+.2f}mm")
                review = review or not override
                if dev > 0 and not override:
                    warnings.extend(_inner_hints(entry, ("left", "right") if axis == "width" else ("top", "bottom"),
                                                 dev * scale.px_per_mm, 0.15 * scale.px_per_mm))
    for side in (entry.get("detection") or {}).get("repaired", []):
        warnings.append(f"edge_repaired:{side}")  # informational: the outer candidate was the holder rim
    if not override:
        edges = (entry.get("detection") or {}).get("edges", {})
        for side, st in edges.items():
            if st.get("n_points") and st.get("n_inliers", st["n_points"]) / st["n_points"] < rules.min_inlier_share:
                warnings.append(f"edge_uncertain:{side}:{st['n_inliers']}/{st['n_points']}")
                review = True
            if st["mad_px"] / scale.px_per_mm > rules.edge_mad_mm:
                warnings.append(f"edge_noisy:{side}")
                review = True
            inside = st.get("inside_px")
            if inside is not None and inside / scale.px_per_mm > margin_mm:
                warnings.append(f"edge_inside:{side}:{inside / scale.px_per_mm:.2f}mm")
                review = True
            if abs(st["bow_px"]) / scale.px_per_mm > rules.edge_bow_mm:
                warnings.append(f"edge_curved:{side}:{st['bow_px'] / scale.px_per_mm:+.2f}mm")
                review = True
    if any(w.startswith("edge_curved") for w in warnings):
        warnings.append("hint:edge flattening (opt-in, planned) may help")
    return ("needs_review" if review else "ok"), warnings, fmt


def result_record(manifest: Manifest, key: str, exported: bool = False) -> dict:
    e = manifest.get(key) or {}
    det = e.get("detection") or {}
    outputs = {k: str(manifest.abs(v)) for k, v in (e.get("outputs") or {}).items()}
    quad = Manifest.effective_quad(e)
    return {
        "type": "result",
        "file": key,
        "input": e.get("input"),
        "status": e.get("status", "pending"),
        "warnings": e.get("warnings", []),
        "quad": None if quad is None else np.round(quad, 1).tolist(),
        "quad_source": "override" if e.get("override") else ("auto" if det else None),
        "size_px": None if quad is None else [round(v, 1) for v in quad_size(quad)],
        "scale": e.get("scale"),
        "format": e.get("format"),
        "edges": det.get("edges"),
        "outputs": outputs,
        "exported": exported,
        "error": e.get("error"),
    }


# --------------------------------------------------------------------------- driver
def _history_scales(manifest: Manifest) -> Dict[Tuple[int, int], Scale]:
    """Median scale per scan-size group from scans already in the manifest (provisional)."""
    return {g: batch_median_scale(s) for g, s in _sizes_by_group(manifest).items() if len(s) >= 3}


def _export_settings(settings: Settings, out_dir: Path) -> dict:
    return {"margin_mm": settings.margin_mm, "interpolation": settings.interpolation,
            "tiff_compression": settings.tiff_compression, "overlay": settings.overlay,
            "derivatives": [asdict(d) for d in settings.derivatives], "out_dir": str(out_dir)}


def _workers_for(manifest: Manifest, jobs: List[dict], settings: Settings) -> int:
    meta = None
    for j in jobs:
        meta = (manifest.get(j["key"]) or {}).get("meta")
        if meta:
            break
    if not meta:
        m = read_meta(jobs[0]["path"])
        meta = {"width": m.width, "height": m.height, "bits": m.bits}
    return auto_workers(len(jobs), bytes_per_worker(meta["width"], meta["height"], meta["bits"]), settings.workers)


def _store_job_result(manifest: Manifest, res: dict) -> None:
    key = res["key"]
    e = manifest.entry(key)
    if res.get("fingerprint"):
        if e.get("fingerprint") and e["fingerprint"] != res["fingerprint"] and e.get("override"):
            manifest.set_override(key, None, "system", "input content changed; manual fix dropped", "reset")
        e["fingerprint"] = res["fingerprint"]
    for k in ("meta", "size_bytes", "mtime"):
        if k in res:
            e[k] = res[k]
    e["input"] = str(Path(res["path"]).resolve())
    if "detection" in res:
        e.update(detection=res["detection"], detected_at=now())
    if "error" in res:
        if res["error"].startswith("detection failed"):
            e["detection"] = None
        e.update(status="failed", error=res["error"], warnings=[])
        return
    e["error"] = None
    if "outputs" in res:
        e.update(outputs={n: manifest.rel(p) for n, p in res["outputs"].items()},
                 output_ppm=res["output_ppm"], output_size_px=res["output_size_px"],
                 export_px_per_mm=res["export_px_per_mm"], exported_at=now())


def run_batch(inputs: Sequence[str], out_dir: Optional[str], settings: Settings,
              emit: Callable[[dict], None] = lambda r: None, recursive: bool = False,
              progress: Optional[Callable[[str, str], None]] = None) -> dict:
    """Process scans into ``out_dir``. Emits one result record per scan and returns a summary.

    ``progress(key, what)`` is called as each worker job finishes ("detected", "exported" or
    an error message); scans that are up to date produce no call.

    Each scan is decoded once: detection (if the scan is new or changed) and export run in
    the same worker with a provisional scale. Once every scan is measured, the final scale
    is resolved per scan-size group; outputs whose scale changed get their resolution
    metadata patched, or are re-exported when the crop margin would move by >= 0.5 px.
    """
    t0 = time.perf_counter()
    files = collect_inputs(inputs, recursive)
    if not files:
        raise FileNotFoundError("no .tif/.tiff scans found in the given inputs")
    names = [scan_key(str(f)) for f in files]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate scan file names (outputs would collide): {sorted(dupes)}")
    if out_dir:
        out_res = Path(out_dir).resolve()
        for f in files:
            if f.resolve().parent == out_res:
                raise ValueError("output folder must differ from the input folder")
    manifest = Manifest(out_dir, persist=bool(out_dir) and not settings.dry_run)
    out_path = manifest.out_dir
    manifest.data["settings"] = settings_to_dict(settings)
    keys = names
    explicit = explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    history = _history_scales(manifest)
    ex_settings = _export_settings(settings, out_path)
    export_now = manifest.persist and not settings.strict  # strict needs the final status first
    exported_now: Dict[str, bool] = {}

    # ---- pass A: detect new or changed scans, export them from the same decode ----------
    jobs = []
    for f, key in zip(files, keys):
        e = manifest.get(key) or {}
        st = os.stat(f)
        unchanged = (e.get("detection") and e.get("size_bytes") == st.st_size
                     and abs(e.get("mtime", 0) - st.st_mtime) < 1e-3)
        if unchanged and not settings.redetect:
            continue
        prov = explicit or (history.get(_group(e)) if e else None)
        jobs.append({"key": key, "path": str(f), "detect": True, "detection": None,
                     "override": None, "export": export_now, "scale": prov.px_per_mm if prov else None,
                     "export_settings": ex_settings, "detect_params": settings.detect_params})
    if jobs:
        workers = _workers_for(manifest, jobs, settings)
        log.info("processing %d new or changed scan(s) with %d worker(s)", len(jobs), workers)
        for res in _run_pool(_job, jobs, workers):
            _store_job_result(manifest, res)
            exported_now[res["key"]] = "outputs" in res
            log.info("%s: %s in %.1fs", res["key"], res.get("error") or "done", res["time_s"])
            if progress:
                progress(res["key"], res.get("error") or "detected")
            manifest.save()

    # ---- scale and review --------------------------------------------------------------
    scales = resolve_scales(manifest, keys, settings)
    refs = reference_sizes(manifest)
    for k in keys:
        e = manifest.entry(k)
        if k in scales and (e.get("detection") or e.get("override")):
            status, warnings, fmt = evaluate(e, scales[k], settings.rules, refs.get(_group(e)), settings.margin_mm)
            e.update(scale=scales[k].to_dict(), format=fmt, status=status, warnings=warnings, error=None)

    # ---- reconcile exports with the final scale ---------------------------------------
    pass_b = []
    for k in keys:
        e = manifest.entry(k)
        if e.get("status") == "failed" or k not in scales or not manifest.persist:
            continue
        if settings.strict and e.get("status") == "needs_review":
            continue
        quad = Manifest.effective_quad(e)
        final = scales[k]
        xkey = settings.export_key(quad, final)
        outs = e.get("outputs") or {}
        missing = not outs or any(not manifest.abs(p).exists() for p in outs.values())
        if exported_now.get(k):
            px_p = e.get("export_px_per_mm") or final.px_per_mm
            if abs(px_p - final.px_per_mm) * settings.margin_mm < 0.5:
                if abs(px_p - final.px_per_mm) > 1e-6:
                    ppm = e.get("output_ppm") or {}
                    for name, rel in outs.items():
                        if name in ppm:
                            set_resolution(str(manifest.abs(rel)), final.px_per_mm * ppm[name] / px_p)
                    e["output_ppm"] = {n: final.px_per_mm * v / px_p for n, v in ppm.items()}
                    e["export_px_per_mm"] = final.px_per_mm
                e["export_key"] = xkey
                continue
        elif not (settings.force or e.get("export_key") != xkey or missing):
            continue
        pass_b.append({"key": k, "path": e["input"], "detect": False, "detection": e.get("detection"),
                       "override": e["override"]["quad"] if e.get("override") else None, "export": True,
                       "scale": final.px_per_mm, "export_settings": ex_settings,
                       "detect_params": settings.detect_params, "xkey": xkey})
    if pass_b:
        workers = _workers_for(manifest, pass_b, settings)
        log.info("exporting %d scan(s) with %d worker(s)", len(pass_b), workers)
        xkeys = {j["key"]: j["xkey"] for j in pass_b}
        for res in _run_pool(_job, pass_b, workers):
            _store_job_result(manifest, res)
            e = manifest.entry(res["key"])
            if "error" in res:
                e["error"] = f"export failed: {res['error']}"
            else:
                e["export_key"] = xkeys[res["key"]]
                exported_now[res["key"]] = True
            log.info("%s: exported in %.1fs", res["key"], res["time_s"])
            if progress:
                progress(res["key"], res.get("error") or "exported")
            manifest.save()

    for k in keys:
        emit(result_record(manifest, k, exported=exported_now.get(k, False)))
    counts: Dict[str, int] = {}
    for k in keys:
        st = manifest.entry(k).get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    manifest.save()
    return {
        "type": "summary",
        "scans": len(keys),
        "counts": counts,
        "exported": sum(1 for k in keys if exported_now.get(k)),
        "needs_review": sorted(k for k in keys if manifest.entry(k).get("status") == "needs_review"),
        "failed": sorted(k for k in keys if manifest.entry(k).get("status") == "failed"),
        "manifest": str(manifest.path) if manifest.persist else None,
        "time_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- settings persistence
PERSISTED = ("margin_mm", "interpolation", "tiff_compression", "overlay", "ppi", "profile", "profiles_file", "strict")


def settings_to_dict(s: Settings) -> dict:
    d = {k: getattr(s, k) for k in PERSISTED}
    d["derivatives"] = [asdict(x) for x in s.derivatives]
    d["rules"] = asdict(s.rules)
    return d


def settings_from_dict(d: dict) -> Settings:
    s = Settings()
    for k in PERSISTED:
        if k in d:
            setattr(s, k, d[k])
    if "derivatives" in d:
        s.derivatives = tuple(Derivative(**x) for x in d["derivatives"])
    if "rules" in d:
        s.rules = ReviewRules(**d["rules"])
    return s
