"""Batch processing: detect, export, resolve scale, review. Resumable via the manifest."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from . import flatten as flat
from . import refit
from . import shading
from .detection import DetectionError, DetectParams, band_shift, detect, quad_size
from .formats import FORMATS, FORMAT_OF_ITEM, classify, match_format
from .geometry import rectify, shift_side
from . import metadata as meta_mod
from .item import (AGENT_WARNING, SCALE_UNVERIFIED, agent_crop, forced_format, held_by_hand, is_other, item_of,
                   kind_of, scan_ppi_of, valid_item)
from .imageio import (SCAN_SUFFIXES, file_fingerprint, load_rgb, read_meta, resize_long_edge, set_resolution,
                      write_derivative, write_tiff)
from .meta_patch import PatchError, set_metadata
from .metadata import FileMeta
from .lock import RunLock
from .manifest import MANIFEST_NAME, Manifest, now, scan_key
from .preview import LONG_EDGE, PREVIEW_DIR, write_preview
from .render import overlay
from .scale import (MM_PER_INCH, NO_SCALE, SCAN_PPI, TAG_UNVERIFIED, Scale, batch_median_scale, explicit_scale,
                    scale_from_size, scale_from_tag, scale_from_tag_as_is)
from .scan_window import (accept_valid, auto_valid, effective_window, fix_outside, fix_window, outline_id,
                          window_current, window_for_scan, window_shape, window_state)
from .window import inset_quad

log = logging.getLogger("polaroid_processor")

GB = 1024 ** 3

PHOTO_CROP = ("off", "derivatives", "tiff")  # photo-area crop next to the full print: none, JPG derivatives, + TIFF master
PHOTO_BLEED_MM = 0.3
PHOTO_DIR = "photo"
METADATA_SKIPPED = "metadata_not_updated"      # scan warning: a file's headers could not be rewritten (retried next run)
SOFTWARE = f"polaroid-processor {__version__}"      # the Software tag of every export


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
    rim_refit_mm: float = 0.15  # an inner line must explain an oversize this closely to be applied


@dataclass
class Settings:
    margin_mm: float = 0.1
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
    export: bool = True  # False: detect and store the outlines in the manifest, export nothing
    previews: bool = False  # write a small straightened preview per newly detected scan (_previews/)
    workers: Optional[int] = None
    rules: ReviewRules = field(default_factory=ReviewRules)
    detect_params: DetectParams = field(default_factory=DetectParams)
    flatten: str = "auto"  # "auto": unwarp curled prints along their traced edges; "off": straight sides
    shadows: str = "auto"  # "auto": remove edge shadows on plain frames; "off"
    flatten_params: flat.FlattenParams = field(default_factory=flat.FlattenParams)
    shading_params: shading.ShadingParams = field(default_factory=shading.ShadingParams)
    photo_crop: Optional[str] = "off"        # None only from api.make_settings: run_batch fills in the manifest's value
    photo_bleed_mm: Optional[float] = PHOTO_BLEED_MM
    # the project's metadata (see metadata.py); not part of the export key. From api.make_settings: only the fields
    # the caller gave (None: none), merged over the manifest's by run_batch
    metadata: Optional[dict] = None

    def export_key(self, quad: np.ndarray, scale: Scale, curves: Optional[dict] = None,
                   override: bool = False, window: Optional[dict] = None, item: Optional[dict] = None) -> str:
        """Identity of an export: outline, curves, settings, scale and (with a photo crop) the window. ``item``: the
        scan's item choice (``item_fix``); part of the key only when there is one."""
        return self._key(quad, curves, override, round(scale.px_per_mm, 4), window=window, item=item)

    def export_base(self, quad: np.ndarray, scale: Scale, curves: Optional[dict] = None, override: bool = False,
                    window: Optional[dict] = None, item: Optional[dict] = None) -> str:
        """The export key without the scale value: two exports with the same base differ only by the scale
        a batch inferred (its median moved). An explicit scale (``ppi``, a profile, a TIFF tag) is part of it,
        so a deliberate change of scale is always a new export. An item of kind Other has no scale in its base at
        all (the scale only sets its resolution tag)."""
        inferred = scale.source in ("batch-median", "single-scan")
        if valid_item(item) and item.get("kind") == "other":
            return self._key(quad, curves, override, None, None, window, item, base=True)
        return self._key(quad, curves, override, None,
                         ["inferred" if inferred else scale.source, None if inferred else self.ppi,
                          None if inferred else self.profile], window, item)

    def _key(self, quad: np.ndarray, curves: Optional[dict], override: bool, px_per_mm: Optional[float],
             scale_kind: Optional[list] = None, window: Optional[dict] = None, item: Optional[dict] = None,
             base: bool = False) -> str:
        if valid_item(item) and item.get("kind") == "other":
            # exported exactly as drawn: margin, flatten, shadows, curves and the photo crop do not enter it
            payload = {"v": __version__, "derivatives": [asdict(d) for d in self.derivatives],
                       "interpolation": self.interpolation, "tiff_compression": self.tiff_compression,
                       "overlay": self.overlay, "quad": np.round(quad, 2).tolist(),
                       "item": [item["kind"], None if base else item.get("scan_ppi")]}      # the base: the ppi only retags
            if px_per_mm is not None:
                payload["px_per_mm"] = px_per_mm
            return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        payload = {
            "v": __version__,
            "margin_mm": self.margin_mm,
            "derivatives": [asdict(d) for d in self.derivatives],
            "interpolation": self.interpolation,
            "tiff_compression": self.tiff_compression,
            "overlay": self.overlay,
            "quad": np.round(quad, 2).tolist(),
            "flatten": self.flatten,
            "shadows": self.shadows,
        }
        if self.photo_crop != "off":      # absent when off: the keys of exports made before stay valid
            payload["photo"] = [self.photo_crop, self.photo_bleed_mm, window_shape(window)]
        if valid_item(item):      # absent without a choice: the keys of exports made before stay valid
            payload["item"] = [item["kind"], None]
        if px_per_mm is not None:
            payload["px_per_mm"] = px_per_mm
        if scale_kind is not None:
            payload["scale_kind"] = scale_kind
        if self.flatten != "off" and curves and not override:
            payload["curves"] = {s: [round(v, 3) for v in c["coef"]] for s, c in sorted(curves.items())}
            payload["warp"] = [self.flatten_params.fade_mm, self.flatten_params.end_skip_mm, "quad-base"]
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
def manual_outline(override_quad, detection: Optional[dict]) -> bool:
    """True when a reviewer moved the outline; an accepted automatic outline is not manual."""
    if override_quad is None:
        return False
    auto = (detection or {}).get("quad")
    return auto is None or not np.allclose(np.asarray(override_quad, float), np.asarray(auto, float), atol=0.5)


def flatten_plan(curves: Optional[dict], px_per_mm: float, mode: str, override: bool,
                 params: flat.FlattenParams = flat.FlattenParams()) -> flat.FlattenPlan:
    """Whether the export unwarps along curved sides. ``override`` = the outline was moved by a
    reviewer (see :func:`manual_outline`); such outlines stay straight."""
    if mode == "off" or override:
        return flat.FlattenPlan(False, [], {}, {})
    return flat.plan(flat.curves_from_dict(curves), px_per_mm, params)


def _write_images(img: np.ndarray, out_dir: Path, stem: str, px_per_mm: Optional[float], icc: Optional[bytes], ex: dict,
                  prefix: str = "", alpha: Optional[np.ndarray] = None, tiff: bool = True,
                  fm: Optional[FileMeta] = None) -> Tuple[dict, dict]:
    """TIFF master (``out_dir/STEM.tif``) and the derivatives (``out_dir/NAME/STEM.EXT``) of ``img``.
    Returns outputs and their px/mm, named ``prefix`` + ``tiff`` / the derivative's name. ``px_per_mm`` None (an item
    of kind Other without a usable resolution): no resolution tag, and no px/mm in the second result."""
    outputs, ppm = {}, {}
    if tiff:
        tiff_path = out_dir / f"{stem}.tif"
        write_tiff(str(tiff_path), img, px_per_mm, icc, ex["tiff_compression"], alpha=alpha, meta=fm)
        outputs[prefix + "tiff"] = str(tiff_path)
        if px_per_mm:
            ppm[prefix + "tiff"] = px_per_mm
    for d in ex["derivatives"]:
        small = resize_long_edge(img, d["long_edge"], d["scale"])
        f = small.shape[1] / img.shape[1]
        p = out_dir / d["name"] / f"{stem}.{d['fmt']}"
        write_derivative(str(p), small, d["fmt"], d["quality"], px_per_mm * f * 25.4 if px_per_mm else None, icc, meta=fm)
        outputs[prefix + d["name"]] = str(p)
        if px_per_mm:
            ppm[prefix + d["name"]] = px_per_mm * f
    return outputs, ppm


def photo_image(rgb: np.ndarray, window: dict, px_per_mm: float, bleed_mm: float,
                interpolation: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The photo crop: the window quad cut ``bleed_mm`` inward on every side, warped straight from the scan
    (one resampling). Round window: also the alpha plane, 0 outside the ellipse inscribed in the crop (a circle
    for a square crop; the drawing of the window, ``render.circle_of``, is the same ellipse)."""
    img = rectify(rgb, inset_quad(np.asarray(window["quad"], float), bleed_mm * px_per_mm), 0.0, interpolation)
    if window.get("kind") != "round":
        return img, None
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((xx + 0.5 - w / 2) / (w / 2), (yy + 0.5 - h / 2) / (h / 2))      # 1 on the ellipse
    cover = np.clip((1.0 - r) * (h + w) / 4 + 0.5, 0.0, 1.0)
    top = 65535 if img.dtype == np.uint16 else 255
    return img, np.round(cover * top).astype(img.dtype)


def remove_photo_files(out_dir: Path, stem: str) -> None:
    """Delete the photo-crop files of one scan (an export that makes none must not leave an older one behind)."""
    base = out_dir / PHOTO_DIR
    if not base.is_dir():
        return
    cands = [base / f"{stem}.tif"] + [d / f"{stem}.{ext}" for d in base.iterdir() if d.is_dir()
                                       for ext in ("jpg", "png")]
    for f in cands:
        try:
            f.unlink()
        except OSError:
            pass


def _export(rgb: np.ndarray, quad: np.ndarray, px_per_mm: Optional[float], icc: Optional[bytes], key: str, ex: dict,
            curves: Optional[dict] = None, override: bool = False, window: Optional[dict] = None,
            fm: Optional[FileMeta] = None, other: bool = False) -> dict:
    """Rectify (or unwarp), remove edge shadows, write master + derivatives (+ overlay), and with a photo
    window and ``photo_crop`` on the photo crop (``photo/``) from the same scan.

    ``other`` (an item of kind Other): exactly as drawn, no margin, no unwarp, no shadow removal, no photo crop; the
    scale only sets the resolution tag and is None when there is no usable one (then the files get no tag).

    Returns outputs, their px/mm and what was applied.
    """
    if other:
        margin, fplan = 0.0, flat.FlattenPlan(False, [], {}, {})
        window = None
    else:
        margin = ex["margin_mm"] * px_per_mm
        fplan = flatten_plan(curves, px_per_mm, ex["flatten"], override, ex["flatten_params"])
    sides = flat.curves_from_dict(curves) if fplan.apply else None
    if sides:
        img = flat.unwarp(rgb, sides, quad, margin, ex["interpolation"], **ex["flatten_params"].warp_px(px_per_mm))
    else:
        img = rectify(rgb, quad, margin, ex["interpolation"])
    shade = {"sides": []}
    if ex["shadows"] != "off" and not other:
        img, shade = shading.correct(img, px_per_mm, ex["shading_params"])
    stem = Path(key).stem
    out_dir = Path(ex["out_dir"])
    outputs, ppm = _write_images(img, out_dir, stem, px_per_mm, icc, ex, fm=fm)
    photo = (ex.get("photo_crop", "off") != "off" and window and window.get("found") and window.get("quad")
             and not other)
    if photo:     # no shadow correction here: it only touches the frame
        crop, alpha = photo_image(rgb, window, px_per_mm, ex["photo_bleed_mm"], ex["interpolation"])
        o, r = _write_images(crop, out_dir / PHOTO_DIR, stem, px_per_mm, icc, ex, "photo_", alpha,
                             tiff=ex["photo_crop"] == "tiff", fm=fm)
        outputs.update(o)
        ppm.update(r)
    if not photo:
        remove_photo_files(out_dir, stem)
    elif ex["photo_crop"] != "tiff":      # a master from an earlier export made with tiff
        (out_dir / PHOTO_DIR / f"{stem}.tif").unlink(missing_ok=True)
    if ex["overlay"]:
        p = out_dir / "_overlays" / f"{stem}.jpg"
        pts = {"curves": flat.outline_points(sides, quad, **ex["flatten_params"].warp_px(px_per_mm))} if sides else None
        write_derivative(str(p), overlay(rgb, quad, points=pts), "jpg", 85, None)
        outputs["overlay"] = str(p)
    return {"outputs": outputs, "output_ppm": ppm, "output_size_px": [int(img.shape[1]), int(img.shape[0])],
            "flatten": fplan.to_dict(), "shadows": shade}


def file_meta(project_meta: Optional[dict], key: str, input_path: str, fmt: Optional[dict],
              size_px: Optional[Tuple[float, float]], make: str = "", model: str = "",
              unknown_format: str = "", item: Optional[dict] = None) -> FileMeta:
    """What an export of scan ``key`` carries: the project's metadata with the placeholders filled, the scanner's
    Make and Model, and our Software string. ``fmt`` is the scan's format record (its shape decides when it has none).
    ``unknown_format`` stands for ``{format}`` when neither is known (only the app's preview of a scan not yet
    detected uses it; an export always knows). ``item``: the scan's item choice; its kind decides ``{format}``
    (``600`` / ``Go``, empty for Other) and the shape is not asked."""
    kind = (valid_item(item) or {}).get("kind")
    if kind == "other":
        label = ""
    elif kind:
        label = meta_mod.FORMAT_LABELS[FORMAT_OF_ITEM[kind].name]
    else:
        label = meta_mod.format_label(fmt, size_px) or unknown_format
    return meta_mod.fill(project_meta, stem=Path(key).stem, project=Path(input_path).resolve().parent.name,
                         format_name=label, make=make, model=model, software=SOFTWARE)


def sync_metadata(manifest: Manifest, e: dict, key: str, settings: Settings,
                  on_update: Optional[Callable[[], None]] = None, skip_reason: Optional[List[str]] = None) -> str:
    """Bring the headers of a scan's exported files in line with the project's metadata, in place (no pixel is
    touched). ``same``: nothing to do (also when no metadata is set and the export has no metadata key), ``updated``:
    rewritten, ``failed``: a file is not the structure the writers expect, so the scan has to be exported again,
    ``skipped``: a file could not be opened for writing (a viewer holds it); the old key stays and the next run
    tries again (exporting again would fail on the same file; the reason is appended to ``skip_reason``).
    ``on_update`` is called before the first rewrite."""
    quad = Manifest.effective_quad(e)
    fm = file_meta(settings.metadata, key, e["input"], e.get("format"), None if quad is None else quad_size(quad),
                   item=e.get("item_fix"))
    want, have = fm.key(), e.get("metadata_key") or ""
    if want == have:
        return "same"
    names = [n for n in (e.get("outputs") or {}) if n != "overlay"]
    if not names:
        return "failed"
    if on_update:
        on_update()
    try:
        try:
            m = read_meta(e["input"])
            fm = replace(fm, make=m.make, model=m.model)
        except Exception:  # noqa: BLE001  (the scan is gone: the files get no Make and Model)
            pass
        for n in names:
            set_metadata(str(manifest.abs(e["outputs"][n])), fm)
    except OSError as ex:
        log.warning("%s: headers could not be rewritten (%s), the next run tries again", key, ex)
        if skip_reason is not None:
            what = ex.strerror or str(ex)
            skip_reason.append(f"{Path(str(ex.filename)).name}: {what}" if ex.filename else what)
        return "skipped"
    except Exception as ex:  # noqa: BLE001  (PatchError and anything unexpected)
        log.warning("%s: headers could not be rewritten (%s: %s), exporting again", key, type(ex).__name__, ex)
        return "failed"
    if want:
        e["metadata_key"] = want
    else:
        e.pop("metadata_key", None)
    return "updated"


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
        if job["detect"] and job.get("fingerprint") and job["fingerprint"] != res.get("fingerprint"):
            # the file is not the one the crop and the choices were made on (the batch drops them when it stores this
            # result): cut what the file holds now, with the detected outline, as a print
            own_ppi = (valid_item(job.get("item")) or {}).get("scan_ppi")
            job = {**job, "override": None, "item": None, "window_fix": None}
            if own_ppi:      # the scan's own ppi went with its item: the scale a print has stands (explicit, else guessed)
                job["scale"] = job.get("scale_base")
        quad = np.array(job["override"] if job.get("override") else det["quad"], dtype=np.float64)
        ex = job["export_settings"]
        item = valid_item(job.get("item"))
        other = bool(item) and item["kind"] == "other"
        fmt = FORMAT_OF_ITEM.get(item["kind"]) if item else None      # the format the user named, else the shape decides
        if other:      # the scale only sets the resolution tag: provisional, else the TIFF tag as it is, else none
            px = job.get("scale") or None
            if px is None:
                px = scale_from_tag_as_is(meta.dpi_tag).px_per_mm or None
        elif job.get("scale"):
            px = job["scale"]
        else:
            s = (scale_from_tag(meta.dpi_tag, quad_size(quad), (fmt,) if fmt else FORMATS)
                 or scale_from_size(quad_size(quad), fmt))
            px = s.px_per_mm
        if job["detect"] and ex["flatten"] != "off" and det.get("background", "dark") == "dark":
            # a detection must not depend on an item Other (it is traced too, and "auto" means as detected): Other is traced at
            # the scale the scan would have without it (explicit, else the tag or the shape's guess), never at its own ppi or
            # tag as it is; a forced 600 / Go keeps the scale of its format (``px``), as the cut and the window use it
            if item and item["kind"] == "other":
                base = job.get("scale_base", job.get("scale"))
                trace_px = base or (scale_from_tag(meta.dpi_tag, quad_size(quad)) or scale_from_size(quad_size(quad))).px_per_mm
            else:
                trace_px = px
            curves = flat.trace(rgb, np.array(det["quad"], dtype=np.float64), trace_px, det.get("speck_threshold") or 30,
                                ex["flatten_params"])
            det["curves"] = {k: c.to_dict() for k, c in curves.items()}
        window = None
        if ex.get("photo_crop", "off") != "off" and not other:
            fix = job.get("window_fix")
            # the automatic window of this outline (a fix landed after the window pass, or that pass lost the scan);
            # the user's choice, when it holds for the outline, is cut instead. The batch never stores the choice.
            auto = job.get("window")
            if not auto_valid(auto, quad, fmt):
                auto, res["window_time_s"] = _window_step(rgb, quad, fmt)
                res["window"] = auto
            window = effective_window({"window": auto, "window_fix": fix}, quad)
        if job.get("preview"):
            try:  # a preview that fails must not fail the scan
                pv = {**job["preview"], "margin_mm": 0.0, "flatten": "off"} if other else job["preview"]
                res["preview"] = write_preview(rgb, quad, px or 1.0, (det or {}).get("curves"),
                                               manual_outline(job.get("override"), det), pv,
                                               Path(job["preview"]["dir"]) / f"{Path(job['key']).stem}.jpg")
            except Exception as e:  # noqa: BLE001
                log.warning("preview of %s failed: %s", job["key"], e)
        if job["export"]:
            fm = file_meta(ex.get("metadata"), job["key"], path, job.get("format"), quad_size(quad), meta.make, meta.model,
                           item=item)
            res.update(_export(rgb, quad, px, meta.icc_profile, job["key"], ex,
                               curves=(det or {}).get("curves"),
                               override=manual_outline(job.get("override"), det), window=window, fm=fm, other=other))
            res["metadata_key"] = fm.key()
            res["export_px_per_mm"] = px
            res["export_outline"] = outline_id(quad)
            res["export_kind"] = item["kind"] if item else None      # what the files were cut as: the stamp needs it
            res["export_window"] = window      # what the photo crop was cut from: the key and the reconcile need it
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        res["time_s"] = round(time.perf_counter() - t0, 2)
    return res


def _window_step(rgb: np.ndarray, quad: np.ndarray, fmt=None) -> Tuple[dict, float]:
    """The photo window of a scan for the manifest; a failure of the step is a window not found, never an error.
    ``fmt``: the format the user named for the print (item kind 600 or Go), else the shape decides."""
    try:
        return window_for_scan(rgb, quad, fmt=fmt)
    except Exception as e:  # noqa: BLE001
        log.warning("window step failed: %s", e)
        failed = {"found": False, "reason": f"window step failed: {type(e).__name__}: {e}", "kind": None,
                  "format": None, "notes": [], "outline": outline_id(quad), "source": "auto", "accept_needed": True}
        if fmt is not None:
            failed["named"] = True
        return failed, 0.0


def _window_job(job: dict) -> dict:
    """Only the window step of a detected scan (the window is missing or its outline changed)."""
    t0 = time.perf_counter()
    res = {"key": job["key"], "path": job["path"]}
    try:
        res["window"], res["window_time_s"] = _window_step(load_rgb(job["path"]), np.array(job["quad"], dtype=np.float64),
                                                           FORMAT_OF_ITEM.get(job.get("kind") or ""))
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    res["time_s"] = round(time.perf_counter() - t0, 2)
    return res


def _crashed(job: dict) -> dict:
    return {"key": job["key"], "path": job["path"], "error": "worker process crashed on this scan"}


def _kill_pool(ex: ProcessPoolExecutor) -> None:
    """Stop the worker processes at once (there is no public way: the jobs in flight are lost).

    The executor's feeder thread may be blocked writing a queued job into the pipe of a worker that no longer
    reads it, and the executor then waits for that thread for ever. So the reading end is closed too (the writer
    fails and ends) and the executor is told not to wait for the thread.
    """
    if hasattr(ex, "terminate_workers"):      # Python 3.14 and later
        try:
            ex.terminate_workers()
        except Exception:  # noqa: BLE001
            pass
    for proc in list((getattr(ex, "_processes", None) or {}).values()):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    q = getattr(ex, "_call_queue", None)
    if q is not None:
        try:
            q.cancel_join_thread()
            q._reader.close()
        except Exception:  # noqa: BLE001
            pass


def _run_pool(fn: Callable, jobs: List[dict], workers: int, cancel: Optional[Callable[[], bool]] = None):
    """Run ``fn`` on every job in worker processes, yielding results as they finish.

    Only about ``workers`` jobs (plus two queued) are submitted at a time. When ``cancel()`` turns true (checked
    four times a second) the results that are already in are yielded, the jobs not finished are dropped, and the
    worker processes are terminated: the run stops within a moment, not after the scans in flight. A job that was
    cut off leaves no result; its output files are written to a temporary name and renamed when complete, so no
    partial file carries a final name.

    A worker that dies (a native crash in a decoder or OpenCV) breaks the whole pool. The
    jobs left over then run one by one, each in a fresh process, so only the scan that
    crashes fails (as an error result) and the batch goes on.
    """
    if workers <= 1 or len(jobs) <= 1:
        workers = 1
    done = set()
    try:
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            queue = iter(enumerate(jobs))
            pending = {}

            def fill():
                while len(pending) < workers + 2:
                    nxt = next(queue, None)
                    if nxt is None:
                        return
                    pending[ex.submit(fn, nxt[1])] = nxt[0]

            fill()
            while pending:
                finished, _ = wait(list(pending), timeout=0.25, return_when=FIRST_COMPLETED)
                if cancel and cancel():
                    for fut in finished:
                        if not fut.cancelled() and fut.exception() is None:
                            done.add(pending.pop(fut))
                            yield fut.result()
                    _kill_pool(ex)
                    return
                for fut in finished:
                    i = pending.pop(fut)
                    res = fut.result()
                    done.add(i)
                    yield res
                fill()
            return
        finally:
            ex.shutdown(wait=True)
    except BrokenProcessPool:
        log.warning("a worker process crashed; running the remaining scans one by one")
    for i, j in enumerate(jobs):
        if i in done:
            continue
        if cancel and cancel():
            break
        try:
            with ProcessPoolExecutor(max_workers=1) as ex:
                res = ex.submit(fn, j).result()
        except BrokenProcessPool:
            res = _crashed(j)
        yield res


PARTIAL_SUFFIXES = tuple(f"{ext}{tail}" for ext in (".tif", ".tiff", ".jpg", ".jpeg", ".png")
                         for tail in (".part", ".patch", ".part.patch"))      # export, header rewrite, header write


def _clean_partial(out: Path) -> None:
    """Remove the temporary files a killed run left (exports are written to ``NAME.tif.part`` and so on, and
    renamed). Only the names the export writers make are removed, not any ``*.part`` (a browser's partial download
    has that name too). The previews are left alone: the app writes them without the run lock."""
    if not out.is_dir():
        return
    dirs = [out] + [x for x in out.iterdir() if x.is_dir() and x.name != "_previews"]
    if (out / PHOTO_DIR).is_dir():
        dirs += [x for x in (out / PHOTO_DIR).iterdir() if x.is_dir()]
    for d in dirs:
        for f in d.iterdir():
            if f.is_file() and f.name.lower().endswith(PARTIAL_SUFFIXES):
                try:
                    f.unlink()
                except OSError:
                    pass


# --------------------------------------------------------------------------- evaluation
def _group(e: dict) -> Tuple[int, int]:
    """Scans of the same pixel size come from the same scanner setting."""
    m = e.get("meta") or {}
    return m.get("width", 0), m.get("height", 0)


def _sizes_by_group(manifest: Manifest) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """Print sizes per scan-size group (an item of kind Other is no print and is left out)."""
    return {g: [s for s, _ in v] for g, v in _prints_by_group(manifest).items()}


def _prints_by_group(manifest: Manifest) -> Dict[Tuple[int, int], List[Tuple[Tuple[float, float], Optional[object]]]]:
    """``(size, format the user named or None)`` of every print with an outline, per scan-size group. An item of
    kind Other is no print: it never enters the batch scale or the reference sizes. A print the user told the format
    of (item kind 600 or Go) counts like a hand-fixed print, with that format. A crop that still carries the agent
    mark counts for nothing until a human confirms it."""
    out: Dict[Tuple[int, int], list] = {}
    for _, e in manifest.items():
        q = Manifest.effective_quad(e)
        if q is not None and not is_other(e) and not agent_crop(e):
            out.setdefault(_group(e), []).append((quad_size(q), forced_format(e)))
    return out


def _ref_group(e: dict) -> Tuple:
    """Scan-size group and print format (the user's choice, else by shape): the reference size compares like with
    like. An item of kind Other has a group of its own, which no reference size is made for."""
    q = Manifest.effective_quad(e)
    if q is None:
        return _group(e)
    if is_other(e):
        return _group(e) + ("other",)
    return _group(e) + ((forced_format(e) or classify(quad_size(q))).name,)


def reference_sizes(manifest: Manifest, min_scans: int = 5) -> Dict[Tuple, Tuple[float, float]]:
    """Median print size in px per scan-size group and format, for groups with at least
    ``min_scans`` scans."""
    sizes: Dict[Tuple, List[Tuple[float, float]]] = {}
    for _, e in manifest.items():
        q = Manifest.effective_quad(e)
        if q is not None and not is_other(e) and not agent_crop(e):
            sizes.setdefault(_ref_group(e), []).append(quad_size(q))
    return {g: tuple(float(v) for v in np.median(np.array(s), axis=0)) for g, s in sizes.items() if len(s) >= min_scans}


def _side_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _inner_candidates(entry: dict, sides: Tuple[str, str], excess_px: float, tol_px: float) -> List[Tuple[str, float]]:
    """Sides whose inner line (a second line behind the edge) explains an oversized print."""
    edges = (entry.get("detection") or {}).get("edges", {})
    out = []
    for side in sides:
        depth = (edges.get(side) or {}).get("inner_px")
        if depth and abs(depth - excess_px) <= tol_px:
            out.append((side, float(depth)))
    return out


def _inner_hints(entry: dict, sides: Tuple[str, str], excess_px: float, tol_px: float) -> List[str]:
    """Suggest moving a side onto its inner line when that explains an oversized print."""
    return [f"hint:fix --shift {side}=-{depth:.0f} (holder rim parallel to the print)"
            for side, depth in _inner_candidates(entry, sides, excess_px, tol_px)]


def size_refit(entry: dict, scale: Scale, ref_size: Optional[Tuple[float, float]], rules: "ReviewRules",
               load_rgb_fn: Callable[[], np.ndarray]) -> Optional[Tuple[np.ndarray, dict]]:
    """New quad and refit info when the outline sits on a holder rim along one side (see refit.py).

    Only for automatic dark-background detections that are oversized along one axis against
    the batch; ``load_rgb_fn`` is called only when the seam has to be looked up in the scan.
    """
    det = entry.get("detection") or {}
    if (entry.get("override") or not det or det.get("refit") or det.get("background", "dark") != "dark" or not ref_size
            or is_other(entry)):
        return None
    quad = np.array(det["quad"], dtype=np.float64)
    over = refit.oversize(quad, ref_size, rules.size_outlier_mm * scale.px_per_mm)
    if over is None:
        return None
    axis, excess = over
    tol = rules.rim_refit_mm * scale.px_per_mm
    found = refit.from_inner_line(det.get("edges", {}), quad, axis, excess, tol)
    if found is None:
        found = refit.from_seam(load_rgb_fn(), quad, axis, excess, scale.px_per_mm, tol)
    return None if found is None else (found[1], found[2])


def apply_refit(entry: dict, quad: np.ndarray, info: dict, rgb: Optional[np.ndarray], px_per_mm: float,
                settings: "Settings") -> None:
    """Store the refit outline in the detection (the original stays as ``quad_detected``); the
    curves are traced again around it. A refit top or bottom moves on onto the end of its edge
    band, as in detection."""
    det = entry["detection"]
    side = info.get("side")
    if rgb is not None and side in settings.detect_params.band_sides:
        d = band_shift(flat.max_rgb_u8(rgb), quad, side, settings.detect_params, det.get("factor") or 4)
        if d is not None:
            quad = shift_side(quad, side, -d)
            info = {**info, "band_px": round(d, 1)}
    det["quad_detected"] = det["quad"]
    det["quad"] = np.round(quad, 2).tolist()
    det["refit"] = info
    det.pop("curves", None)
    if rgb is not None and settings.flatten != "off":
        curves = flat.trace(rgb, np.array(det["quad"], dtype=np.float64), px_per_mm,
                            det.get("speck_threshold") or 30, settings.flatten_params)
        det["curves"] = {k: c.to_dict() for k, c in curves.items()}


def print_median_scale(manifest: Manifest, entry: dict, prints: Optional[dict] = None) -> Optional[Scale]:
    """The median scale of the project's prints with the same scan size as ``entry`` (at least 3 of them), else None."""
    found = (prints if prints is not None else _prints_by_group(manifest)).get(_group(entry), [])
    if len(found) >= 3:
        return batch_median_scale([s for s, _ in found], fmts=[f for _, f in found])
    return None


def resolve_scales(manifest: Manifest, keys: Sequence[str], settings: Settings) -> Dict[str, Scale]:
    """Scale per scan: explicit > plausible TIFF tag > median of all scans with the same scan size.

    A print the user told the format of (item kind 600 or Go) takes that format wherever the shape would have been
    asked. An item of kind Other has its own order, because its scale only sets the resolution tag: the scan's own
    ppi, the project's ppi or profile, the median of the project's prints with the same scan size (at least 3),
    else the TIFF tag as it is (``tiff-tag-unverified``), else :data:`NO_SCALE` (no tag is written)."""
    explicit = explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    scales: Dict[str, Scale] = {}
    pending: List[str] = []
    prints = _prints_by_group(manifest)

    def group_median(e: dict) -> Optional[Scale]:
        return print_median_scale(manifest, e, prints)

    for k in keys:
        e = manifest.get(k) or {}
        quad = Manifest.effective_quad(e)
        if quad is None:
            continue
        tag = (e.get("meta") or {}).get("dpi_tag")
        if is_other(e):
            own = scan_ppi_of(e)
            scales[k] = (Scale(own / MM_PER_INCH, SCAN_PPI) if own else explicit or group_median(e)
                         or scale_from_tag_as_is(tuple(tag) if tag else None))
            continue
        if explicit:
            scales[k] = explicit
            continue
        fmt = forced_format(e)
        s = scale_from_tag(tuple(tag) if tag else None, quad_size(quad), (fmt,) if fmt else FORMATS)
        if s:
            scales[k] = s
        else:
            pending.append(k)

    for k in pending:
        e = manifest.get(k)
        scales[k] = group_median(e) or scale_from_size(quad_size(Manifest.effective_quad(e)), forced_format(e))
    return scales


def evaluate(entry: dict, scale: Scale, rules: ReviewRules,
             ref_size: Optional[Tuple[float, float]] = None,
             margin_mm: float = 0.1, flatten: str = "auto",
             flatten_params: flat.FlattenParams = flat.FlattenParams(),
             photo_crop: str = "off") -> Tuple[str, List[str], Optional[dict]]:
    """Status and warnings for one scan. ``ref_size`` is the median print size of its scan-size group;
    ``margin_mm`` is the crop margin, which hides edge points that lie at most that far inside.

    When the export unwarps along traced curves, the straight-line checks (``not_parallel``,
    ``edge_inside``, ``edge_curved``) do not apply to the sides that follow a curve.
    """
    warnings: List[str] = []
    quad = Manifest.effective_quad(entry)
    if is_other(entry):
        return _evaluate_other(entry, scale)
    fplan = flatten_plan((entry.get("detection") or {}).get("curves"), scale.px_per_mm, flatten,
                         manual_outline((entry.get("override") or {}).get("quad"), entry.get("detection")),
                         flatten_params)
    curved = set(fplan.sides)
    if fplan.apply:
        warnings.append(f"flattened:{','.join(fplan.sides)}:{max(fplan.bend_mm.values()):.2f}mm")
        for side, reason in fplan.straight.items():
            warnings.append(f"flatten_straight:{side}")  # informational: this side's trace was not reliable
    named = forced_format(entry)      # the format the user told, else the shape decides
    fm = match_format(quad_size(quad), scale.px_per_mm, (named,) if named else FORMATS)
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
        if set(name.split("-")) <= curved:
            continue  # both sides follow their curves
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
    moved = (entry.get("detection") or {}).get("refit")
    if moved:  # informational: the side was moved off the holder rim (size_refit)
        warnings.append(f"rim_refit:{moved['side']}:{moved['shift_px'] / scale.px_per_mm:+.2f}mm")
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
            if side in curved:
                continue  # the export follows this side's traced edge
            if inside is not None and inside / scale.px_per_mm > margin_mm:
                warnings.append(f"edge_inside:{side}:{inside / scale.px_per_mm:.2f}mm")
                review = True
            if abs(st["bow_px"]) / scale.px_per_mm > rules.edge_bow_mm:
                warnings.append(f"edge_curved:{side}:{st['bow_px'] / scale.px_per_mm:+.2f}mm")
                review = True
    if photo_crop != "off":
        # The window in force is the user's (a hand-set window inside the outline, or no photo crop by choice),
        # else the automatic one. A window not found, or a hand-set window the outline moved away from, is a review
        # item until the user sets a window, chooses no photo crop or accepts the scan for this outline (what is
        # cut then stays as it is: no photo crop for a missing window, the automatic window when only the hand-set
        # window is outside and an automatic one was found). A fix of the outline alone does not count. An automatic window
        # made before the accept mark existed (no ``accept_needed``) was accepted by any override.
        fix_out = fix_outside(entry, quad)
        win = entry.get("window")
        missing = bool(win) and window_state(effective_window(entry, quad)) == "missing"
        if fix_out or missing:
            accepted = accept_valid(entry, quad) or (missing and "accept_needed" not in win and bool(override))
            if fix_out:
                warnings.append("photo window set by hand lies outside the print outline")
            if missing:
                warnings.append(f"photo window not found: {win.get('reason') or 'unknown'}")
            review = review or not accepted
    if any(w.startswith("edge_curved") for w in warnings) and flatten == "off":
        warnings.append("hint:process with flatten on to unwarp curved edges")
    if agent_crop(entry):      # an agent placed the crop on a scan without a detected outline: a human looks at it
        warnings.append(AGENT_WARNING)
        review = True
    return ("needs_review" if review else "ok"), warnings, fmt


def _evaluate_other(entry: dict, scale: Scale) -> Tuple[str, List[str], Optional[dict]]:
    """Status and warnings of an item of kind Other: it is exported exactly as drawn, so no size, shape, edge or window
    rule applies and it is ``ok``, except a crop an agent placed (``cropped by agent``: a human confirms or saves it).
    The scale only sets the resolution tag: ``scale_unverified`` says it could not be checked against anything and
    never makes the scan need review."""
    warnings: List[str] = []
    if scale.source == "batch-median":
        warnings.append(f"scale_inferred:{scale.source}")
    if scale.source in (TAG_UNVERIFIED, NO_SCALE.source):
        warnings.append(SCALE_UNVERIFIED)      # with no tag at all the files get no resolution tag
    review = False
    if agent_crop(entry):
        warnings.append(AGENT_WARNING)
        review = True
    return ("needs_review" if review else "ok"), warnings, None


def store_judgement(e: dict, scale: Scale, fmt: Optional[dict], status: str, warnings: List[str]) -> None:
    """Put a scan's status, warnings, scale and format into its entry and clear its error. When a crop was put on a scan
    nothing was detected on (it is no longer failed), the error it had stays as ``detection_error``."""
    err = e.get("error")
    if err and not e.get("detection") and e.get("override"):      # a crop over a scan nothing was detected on
        e["detection_error"] = err
    e.update(scale=scale.to_dict(), format=fmt, status=status, warnings=warnings, error=None)


def reevaluate(manifest: Manifest, keys: Sequence[str], settings: Settings) -> None:
    """Status and warnings of ``keys`` again, from the manifest alone (no image is read, and an outline
    moved by a reviewer is never refitted). Used after a fix; nothing is saved."""
    scales = resolve_scales(manifest, keys, settings)
    refs = reference_sizes(manifest)
    for k in keys:
        e = manifest.get(k)
        if k in scales and e and (e.get("detection") or e.get("override")):
            status, warnings, fmt = evaluate(e, scales[k], settings.rules, refs.get(_ref_group(e)), settings.margin_mm,
                                             settings.flatten, settings.flatten_params, settings.photo_crop)
            store_judgement(e, scales[k], fmt, status, warnings)


class _Saver:
    """Saves the manifest at most every ``every`` seconds while jobs finish (saving costs time in
    proportion to the scans), and judges again what someone else fixed in the meantime."""

    def __init__(self, manifest: Manifest, settings: Settings, every: float = 1.5):
        self.manifest, self.settings, self.every = manifest, settings, every
        self.last = time.monotonic()

    def tick(self) -> None:
        if time.monotonic() - self.last >= self.every:
            self.flush()

    def flush(self) -> None:
        self.manifest.save()
        fixed = self.manifest.take_fixed()
        if fixed:      # a fix from outside: its status must come from the fixed outline, not from this run
            reevaluate(self.manifest, fixed, self.settings)
            self.manifest.save()
        self.last = time.monotonic()


def window_record(e: dict) -> dict:
    """The photo window of a manifest entry as the result records show it."""
    quad = Manifest.effective_quad(e)
    w = effective_window(e, quad) or {}      # the window in force: the user's choice, else the automatic window
    rec = {k: w[k] for k in ("found", "kind", "format", "size_mm", "quad", "reason") if w.get(k) not in (None, "")}
    rec["found"] = bool(w.get("found"))
    rec["state"] = window_state(w)      # found, none (no photo crop by choice) or missing
    rec["source"] = w.get("source", "auto")
    if rec["source"] == "manual":
        rec.update({k: w[k] for k in ("author", "at", "note") if w.get(k)})
    if fix_outside(e, quad):      # a hand-set window the outline moved away from: flagged, not in force; the corners
        hw = fix_window(e)         # are what a window fix edits and `inspect --of window` shows
        rec["outside_outline"] = {k: hw[k] for k in ("kind", "quad", "author", "at", "note") if hw.get(k)}
    if accept_valid(e, quad):
        rec["accepted"] = True
    rec["current"] = window_current(e)
    if w.get("notes"):
        rec["notes"] = w["notes"]
    return rec


def result_record(manifest: Manifest, key: str, exported: bool = False) -> dict:
    e = manifest.get(key) or {}
    det = e.get("detection") or {}
    outputs = {k: str(manifest.abs(v)) for k, v in (e.get("outputs") or {}).items()}
    quad = Manifest.effective_quad(e)
    rec = {
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
        "flatten": e.get("flatten"),
        "shadows": e.get("shadows"),
        "outputs": outputs,
        "exported": exported,
        "error": e.get("error"),
    }
    crop_on = manifest.data.get("settings", {}).get("photo_crop", "off") != "off"
    if (e.get("window") or fix_window(e)) and crop_on and not is_other(e):
        rec["window"] = window_record(e)
    if item_of(e):      # only with a choice: records of scans "as detected" stay as they were
        rec["item"] = item_record(e)
    if agent_crop(e):
        rec["agent_crop"] = True
    if e.get("detection_error") and not det:
        rec["detection_error"] = e["detection_error"]
    return rec


def item_record(e: dict) -> dict:
    """The item choice of a manifest entry as the records show it."""
    it = item_of(e)
    rec = {"kind": it["kind"]}
    if it.get("scan_ppi"):
        rec["scan_ppi"] = it["scan_ppi"]
    rec.update({k: it[k] for k in ("author", "at", "note") if it.get(k)})
    rec["agent_crop"] = bool(agent_crop(e))      # the mark is the entry's own, see item.agent_crop
    return rec


# --------------------------------------------------------------------------- driver
def _history_scales(manifest: Manifest) -> Dict[Tuple[int, int], Scale]:
    """Median scale per scan-size group from scans already in the manifest (provisional)."""
    return {g: batch_median_scale([x for x, _ in s], fmts=[f for _, f in s])
            for g, s in _prints_by_group(manifest).items() if len(s) >= 3}


def _export_settings(settings: Settings, out_dir: Path) -> dict:
    return {"margin_mm": settings.margin_mm, "interpolation": settings.interpolation,
            "tiff_compression": settings.tiff_compression, "overlay": settings.overlay,
            "derivatives": [asdict(d) for d in settings.derivatives], "out_dir": str(out_dir),
            "flatten": settings.flatten, "shadows": settings.shadows,
            "flatten_params": settings.flatten_params, "shading_params": settings.shading_params,
            "photo_crop": settings.photo_crop, "photo_bleed_mm": settings.photo_bleed_mm,
            "metadata": settings.metadata or {}}


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
        if e.get("fingerprint") and e["fingerprint"] != res["fingerprint"]:
            if e.get("override"):
                manifest.set_override(key, None, "system", "input content changed; manual fix dropped", "reset")
            if e.get("window_fix"):      # the choice and the accept mark: both described the old picture
                if fix_window(e):
                    manifest.set_window_fix(key, None, "system", "input content changed; window choice dropped",
                                            "window-drop")
                e["window_fix"] = None
            if e.get("item_fix"):      # the item kind and the scan's own ppi: they described the old picture
                manifest.set_item_fix(key, None, "system", "input content changed; item choice dropped", "item-drop")
            if e.get("agent_crop"):      # an agent's crop of the old picture is gone, and so is its mark
                manifest.set_agent_crop(key, None, "system", "input content changed; agent mark dropped",
                                        "agent-crop-drop")
            e.pop("detection_error", None)      # what failed on the old picture says nothing about this one
        e["fingerprint"] = res["fingerprint"]
    for k in ("meta", "size_bytes", "mtime"):
        if k in res:
            e[k] = res[k]
    e["input"] = str(Path(res["path"]).resolve())
    if "detection" in res:
        e.update(detection=res["detection"], detected_at=now())
        e.pop("detection_error", None)
    if "error" in res:
        if res["error"].startswith("detection failed"):
            e["detection"] = None
            if held_by_hand(e):      # cropped by hand: the crop stays in force, the scan stays ok; why detection failed is kept
                e["detection_error"] = res["error"]
                e["error"] = None
                return
        e.update(status="failed", error=res["error"], warnings=[])
        return
    e["error"] = None
    if "window" in res:
        e["window"] = res["window"]
    if "preview" in res:
        e["preview"] = {**res["preview"], "file": manifest.rel(res["preview"]["file"])}
    if "outputs" in res:
        e.update(outputs={n: manifest.rel(p) for n, p in res["outputs"].items()},
                 output_ppm=res["output_ppm"], output_size_px=res["output_size_px"],
                 export_px_per_mm=res["export_px_per_mm"], export_geom_ppm=res["export_px_per_mm"],
                 exported_at=now(),
                 flatten=res.get("flatten"), shadows=res.get("shadows"))
        if res.get("metadata_key"):
            e["metadata_key"] = res["metadata_key"]
        else:
            e.pop("metadata_key", None)


def retag_ok(e: dict, final: Scale, settings: Settings, curves: Optional[dict] = None, manual: bool = False) -> bool:
    """An export that differs from what a run would write only by an inferred scale can keep its pixels when
    the scale it was cut at is within 0.5 % of the new one (the margin and the widths of the unwarp and the
    shading move by a fraction of a pixel) and the unwarp decision (flatten or not) comes out the same.

    An item of kind Other has no margin, unwarp or shading, so any scale keeps its pixels. Only the tags of an export
    that has them can be patched and only to a scale that exists: an export made without a resolution tag (or
    needing none any more) is exported again."""
    geom = e.get("export_geom_ppm")
    if not geom or not final.known:
        return False
    if is_other(e):
        return True
    if abs(geom - final.px_per_mm) / final.px_per_mm >= 0.005:
        return False
    plan = flatten_plan(curves, final.px_per_mm, settings.flatten, manual, settings.flatten_params)
    return plan.apply == bool((e.get("flatten") or {}).get("apply"))


def retag(manifest: Manifest, e: dict, final: Scale) -> None:
    """Write the resolution of ``final`` into the tags of the outputs, in place (nothing is encoded again)."""
    px_p = e.get("export_px_per_mm") or final.px_per_mm
    if abs(px_p - final.px_per_mm) > 1e-6:
        ppm = e.get("output_ppm") or {}
        for name, rel in (e.get("outputs") or {}).items():
            if name in ppm:
                set_resolution(str(manifest.abs(rel)), final.px_per_mm * ppm[name] / px_p)
        e["output_ppm"] = {n: final.px_per_mm * v / px_p for n, v in ppm.items()}
        e["export_px_per_mm"] = final.px_per_mm


def run_batch(inputs: Sequence[str], out_dir: Optional[str], settings: Settings,
              emit: Callable[[dict], None] = lambda r: None, recursive: bool = False,
              progress: Optional[Callable[[str, str], None]] = None,
              cancel: Optional[Callable[[], bool]] = None,
              on_jobs: Optional[Callable[[int], None]] = None,
              on_phase: Optional[Callable[[str], None]] = None) -> dict:
    """Process scans into ``out_dir``. Emits one result record per scan and returns a summary.

    ``progress(key, what)`` is called as each worker job finishes ("detected", "exported" or
    an error message); scans that are up to date produce no call. ``on_jobs(n)`` adds ``n`` jobs
    to the total: the scans that need detection (before the first one starts; also when 0), then
    the scans to export. ``on_phase(text)`` says what the batch does after the scans are detected
    ("checking sizes", a scan being refitted). When ``cancel()`` turns true, the scans not yet
    started are left out (the summary says ``cancelled``); scans that are not exported yet stay so.

    With ``settings.export`` off the run only detects: outlines, curves, status and warnings go
    into the manifest, no scan is exported, and a later run exports from the stored detections.

    Each scan is decoded once: detection (if the scan is new or changed) and export run in
    the same worker with a provisional scale. Once every scan is measured, the final scale
    is resolved per scan-size group; outputs whose scale changed get their resolution
    metadata patched, or are re-exported when the crop margin would move by >= 0.5 px.

    One batch works on an output folder at a time (:class:`RunLock`; a second one raises
    :class:`RunLocked`). A fix saved by someone else during the run is merged in, not overwritten.
    """
    if out_dir and not settings.dry_run:
        with RunLock(out_dir):
            return _run_batch(inputs, out_dir, settings, emit, recursive, progress, cancel, on_jobs, on_phase)
    return _run_batch(inputs, out_dir, settings, emit, recursive, progress, cancel, on_jobs, on_phase)


def _run_batch(inputs, out_dir, settings, emit, recursive, progress, cancel, on_jobs, on_phase) -> dict:
    t0 = time.perf_counter()
    files = collect_inputs(inputs, recursive)
    if not files:
        raise FileNotFoundError("no .tif/.tiff scans found in the given inputs")
    names = [scan_key(str(f)) for f in files]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(f"duplicate scan file names (outputs would collide): {sorted(dupes)}")
    stems = [Path(n).stem.casefold() for n in names]      # case-folded: A.tif and a.tif collide on Windows and macOS
    shared = sorted(n for n, st in zip(names, stems) if stems.count(st) > 1)
    if shared:      # a.tif next to a.tiff: both write <stem>.tif, <stem>.jpg ...
        raise ValueError(f"scan file names with the same stem (outputs would collide): {shared}")
    if out_dir:
        out_res = Path(out_dir).resolve()
        for f in files:
            if f.resolve().parent == out_res:
                raise ValueError("output folder must differ from the input folder")
    manifest = Manifest(out_dir, persist=bool(out_dir) and not settings.dry_run)
    out_path = manifest.out_dir
    if manifest.persist:
        _clean_partial(out_path)
    stored = manifest.data.get("settings") or {}      # an option left out keeps the project's value
    if settings.photo_crop is None or settings.photo_bleed_mm is None:
        settings = replace(settings, photo_crop=stored.get("photo_crop", "off") if settings.photo_crop is None
                           else settings.photo_crop,
                           photo_bleed_mm=stored.get("photo_bleed_mm", PHOTO_BLEED_MM) if settings.photo_bleed_mm is None
                           else settings.photo_bleed_mm)
    try:      # fields the caller gave replace the project's, an empty one clears it, the others stay
        settings = replace(settings, metadata=meta_mod.merge(stored.get("metadata"), settings.metadata))
    except ValueError:
        try:
            meta_mod.normalize(settings.metadata, keep_empty=True)
        except ValueError:      # a given dict that fails the checks (not from make_settings): read as saved values are
            pass
        else:      # what was given is fine; with the saved fields it is too much: refused before anything is written
            raise
        merged = meta_mod.normalize_stored(stored.get("metadata"))
        merged.update(meta_mod.normalize_stored(settings.metadata))
        settings = replace(settings, metadata=merged)
    manifest.data["settings"] = settings_to_dict(settings)
    keys = names
    explicit = explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    history = _history_scales(manifest)
    ex_settings = _export_settings(settings, out_path)
    export_now = manifest.persist and settings.export and not settings.strict  # strict needs the final status first
    preview = {"dir": str(out_path / PREVIEW_DIR), "long_edge": LONG_EDGE, "margin_mm": settings.margin_mm,
               "flatten": settings.flatten, "flatten_params": settings.flatten_params
               } if manifest.persist and settings.previews else None
    exported_now: Dict[str, bool] = {}
    exp_outline: Dict[str, str] = {}   # outline_id of the outline a scan's export (pass A) was cut from
    exp_window: Dict[str, Optional[list]] = {}   # window_shape of the window its photo crop was cut from
    exp_kind: Dict[str, Optional[str]] = {}      # the item kind its export (pass A) was cut as: None, 600, go or other
    saver = _Saver(manifest, settings)

    # ---- pass A: detect new or changed scans, export them from the same decode ----------
    jobs = []
    for f, key in zip(files, keys):
        e = manifest.get(key) or {}
        st = os.stat(f)
        same_file = e.get("size_bytes") == st.st_size and abs(e.get("mtime", 0) - st.st_mtime) < 1e-3
        # a scan cropped by hand that detection failed on is not detected again (only ``redetect`` does); a changed
        # file is, which also drops the crop and the item choice (see _store_job_result)
        unchanged = same_file and (held_by_hand(e) or bool(
            e.get("detection") and (settings.flatten == "off" or "curves" in e["detection"]
                                    or e["detection"].get("background") == "light")))  # light: never traced
        if unchanged and not settings.redetect:
            continue
        prov = explicit or (history.get(_group(e)) if e else None)
        base_prov = prov      # without the item's own ppi: what a print of this scan would start from
        if is_other(e) and scan_ppi_of(e):      # the scan's own resolution comes first
            prov = Scale(scan_ppi_of(e) / MM_PER_INCH, SCAN_PPI)
        jobs.append({"key": key, "path": str(f), "detect": True, "detection": None,
                     "override": e["override"]["quad"] if e.get("override") else None, "item": e.get("item_fix"),
                     "export": export_now, "scale": prov.px_per_mm if prov else None,
                     "scale_base": base_prov.px_per_mm if base_prov else None,
                     "window": None, "window_fix": e.get("window_fix"),
                     "fingerprint": e.get("fingerprint"),
                     "export_settings": ex_settings, "detect_params": settings.detect_params, "preview": preview})
    if on_jobs:
        on_jobs(len(jobs))
    if jobs:
        workers = _workers_for(manifest, jobs, settings)
        log.info("processing %d new or changed scan(s) with %d worker(s)", len(jobs), workers)
        for res in _run_pool(_job, jobs, workers, cancel):
            _store_job_result(manifest, res)
            exported_now[res["key"]] = "outputs" in res
            if "outputs" in res:
                exp_outline[res["key"]] = res["export_outline"]
                exp_window[res["key"]] = window_shape(res["export_window"])
                exp_kind[res["key"]] = res["export_kind"]
            log.info("%s: %s in %.1fs", res["key"], res.get("error") or "done", res["time_s"])
            if progress:
                progress(res["key"], res.get("error") or "detected")
            saver.tick()
        saver.flush()

    # ---- scale and review --------------------------------------------------------------
    manifest.refresh()   # fixes saved by someone else during detection (every scan is judged below)
    scales = resolve_scales(manifest, keys, settings)
    refs = reference_sizes(manifest)
    if on_phase:
        on_phase("checking sizes")
    last_refresh = time.monotonic()
    ref_of: Dict[str, Optional[Tuple[float, float]]] = {}
    for k in keys:
        e = manifest.entry(k)
        if k in scales and (e.get("detection") or e.get("override")):
            ref = ref_of[k] = refs.get(_ref_group(e))
            status, warnings, fmt = evaluate(e, scales[k], settings.rules, ref, settings.margin_mm,
                                             settings.flatten, settings.flatten_params, settings.photo_crop)
            if status == "needs_review":      # fixed meanwhile? (a refresh parses the file: at most once a second)
                if time.monotonic() - last_refresh >= 1.0:
                    manifest.refresh()
                    last_refresh = time.monotonic()
                if k in manifest.fixed:
                    status, warnings, fmt = evaluate(e, scales[k], settings.rules, ref, settings.margin_mm,
                                                     settings.flatten, settings.flatten_params, settings.photo_crop)
            if status == "needs_review" and not (cancel and cancel()):    # (a refit loads the scan: not after a cancel)
                cache = {}

                def scan_rgb(path=e["input"], cache=cache):
                    if "rgb" not in cache:
                        cache["rgb"] = load_rgb(path)
                    return cache["rgb"]

                if on_phase:
                    on_phase(f"checking {k}")
                fit = size_refit(e, scales[k], ref, settings.rules, scan_rgb)
                if fit:
                    apply_refit(e, fit[0], fit[1], scan_rgb(), scales[k].px_per_mm, settings)
                    status, warnings, fmt = evaluate(e, scales[k], settings.rules, ref, settings.margin_mm,
                                                     settings.flatten, settings.flatten_params, settings.photo_crop)
            store_judgement(e, scales[k], fmt, status, warnings)

    # ---- photo windows: scans whose window is missing or was found for another outline ----
    if settings.photo_crop != "off" and not (cancel and cancel()):
        w_jobs = []
        for k in keys:
            e = manifest.entry(k)
            quad = Manifest.effective_quad(e)
            if (quad is not None and e.get("status") != "failed" and e.get("input") and not is_other(e)
                    and not auto_valid(e.get("window"), quad, forced_format(e))):
                w_jobs.append({"key": k, "path": e["input"], "quad": quad.tolist(), "kind": kind_of(e)})
        if w_jobs:      # not counted as jobs (the app's "exported N scans" counts scans), the phase says what runs
            if on_phase:
                on_phase("finding photo windows")
            log.info("finding the photo window of %d scan(s)", len(w_jobs))
            found = []
            for res in _run_pool(_window_job, w_jobs, _workers_for(manifest, w_jobs, settings), cancel):
                if "window" in res:
                    manifest.entry(res["key"])["window"] = res["window"]
                    found.append(res["key"])
                log.info("%s: photo window %s in %.1fs", res["key"], res.get("error") or "done", res["time_s"])
                saver.tick()
            for k in found:     # judged again with the scale and reference size of the pass above: a window not found is a review item
                e = manifest.entry(k)
                if k in scales and (e.get("detection") or e.get("override")):
                    status, warnings, fmt = evaluate(e, scales[k], settings.rules, ref_of.get(k), settings.margin_mm,
                                                     settings.flatten, settings.flatten_params, settings.photo_crop)
                    store_judgement(e, scales[k], fmt, status, warnings)
            saver.flush()

    # ---- reconcile exports with the final scale ---------------------------------------
    manifest.refresh()
    fixed = manifest.take_fixed()     # fixes made while the sizes were checked, also of scans judged by now
    if fixed:
        reevaluate(manifest, fixed, settings)
        scales.update(resolve_scales(manifest, fixed, settings))      # a scan cropped or re-kinded meanwhile has its own
    pass_b = []
    xargs: Dict[str, tuple] = {}      # quad, scale, curves, manual, item of each export job: for the key of the window it cut
    retagged = metadata_updated = metadata_skipped = 0
    phased = []

    def phase_once() -> None:
        if on_phase and not phased:
            phased.append(1)
            on_phase("updating metadata")
    for k in keys:
        e = manifest.entry(k)
        if e.get("status") == "failed" or k not in scales or not manifest.persist or not settings.export:
            continue
        if cancel and cancel():
            pass_b = []
            break
        if settings.strict and e.get("status") == "needs_review":
            continue
        quad = Manifest.effective_quad(e)
        final = scales[k]
        curves = (e.get("detection") or {}).get("curves")
        manual = manual_outline((e.get("override") or {}).get("quad"), e.get("detection"))
        win = effective_window(e, quad) if settings.photo_crop != "off" else None      # what the photo crop is cut from
        item = e.get("item_fix")
        xkey = settings.export_key(quad, final, curves, manual, win, item)
        xbase = settings.export_base(quad, final, curves, manual, win, item)
        outs = e.get("outputs") or {}
        missing = not outs or any(not manifest.abs(p).exists() for p in outs.values())
        keep = False
        if exported_now.get(k):
            # cut at a scale within 0.5 % of the final one, from the outline, the window and the item kind the scan has
            # now (a refit or a fix merged during the run moved them: the files then are not what the key would say)
            if (exp_outline.get(k) == outline_id(quad) and exp_window.get(k) == window_shape(win)
                    and exp_kind.get(k) == kind_of(e) and retag_ok(e, final, settings, curves, manual)):
                retag(manifest, e, final)
                e["export_key"], e["export_base"] = xkey, xbase
                keep = True
        elif (not settings.force and not missing and e.get("export_key") != xkey and e.get("export_base") == xbase
              and retag_ok(e, final, settings, curves, manual)):
            retag(manifest, e, final)       # only the scale moved (the batch median): tags, no new pixels
            e["export_key"] = xkey
            retagged += 1
            keep = True
        elif not (settings.force or e.get("export_key") != xkey or missing):
            keep = True
        if keep:      # the pixels stay; the headers follow the project's metadata (not part of the export key)
            why: List[str] = []
            state = sync_metadata(manifest, e, k, settings, phase_once, why)
            metadata_updated += state == "updated" and not exported_now.get(k)      # a fresh export is no news
            if state == "skipped":      # visible to the user; the old key stays, so the next run tries again
                metadata_skipped += 1
                note = f"{METADATA_SKIPPED}: {why[0] if why else 'a file could not be written'}"
                e["warnings"] = [w for w in (e.get("warnings") or []) if not w.startswith(METADATA_SKIPPED)] + [note]
            if state != "failed":
                continue
        xargs[k] = (quad, final, curves, manual, item)
        pass_b.append({"key": k, "path": e["input"], "detect": False, "detection": e.get("detection"),
                       "override": e["override"]["quad"] if e.get("override") else None, "export": True,
                       "scale": final.px_per_mm if final.known else None, "export_settings": ex_settings,
                       "window": e.get("window"), "item": item,
                       "window_fix": e.get("window_fix"), "format": e.get("format"),
                       "detect_params": settings.detect_params})
    if pass_b:
        if on_jobs:
            on_jobs(len(pass_b))
        workers = _workers_for(manifest, pass_b, settings)
        log.info("exporting %d scan(s) with %d worker(s)", len(pass_b), workers)
        finished_keys = set()
        for res in _run_pool(_job, pass_b, workers, cancel):
            finished_keys.add(res["key"])
            _store_job_result(manifest, res)
            e = manifest.entry(res["key"])
            if "error" in res:
                e["error"] = f"export failed: {res['error']}"
            else:
                # the key is that of the window the photo was cut from (the job may have cut another one than the
                # reconcile expected: a window fix or a new measurement landed meanwhile)
                *used, item = xargs[res["key"]]
                e["export_key"] = settings.export_key(*used, res["export_window"], item)
                e["export_base"] = settings.export_base(*used, res["export_window"], item)
                if "window" in res:      # measured again in the job
                    if res["key"] in scales and (e.get("detection") or e.get("override")):    # a window lost here is a review item
                        status, warnings, fmt = evaluate(e, scales[res["key"]], settings.rules, ref_of.get(res["key"]),
                                                         settings.margin_mm, settings.flatten, settings.flatten_params,
                                                         settings.photo_crop)
                        store_judgement(e, scales[res["key"]], fmt, status, warnings)
                exported_now[res["key"]] = True
            log.info("%s: exported in %.1fs", res["key"], res["time_s"])
            if progress:
                progress(res["key"], res.get("error") or "exported")
            saver.tick()
        if cancel and cancel():
            for j in pass_b:     # cut off while writing: its files may be half of a new export, so the old key goes
                if j["key"] not in finished_keys:
                    manifest.entry(j["key"]).pop("export_key", None)
                    manifest.entry(j["key"]).pop("metadata_key", None)
        saver.flush()

    saver.flush()      # takes in fixes made during the export: the summary shows their status
    for k in keys:
        emit(result_record(manifest, k, exported=exported_now.get(k, False)))
    counts: Dict[str, int] = {}
    for k in keys:
        st = manifest.entry(k).get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
    saver.flush()
    return {
        "type": "summary",
        "scans": len(keys),
        "counts": counts,
        "exported": sum(1 for k in keys if exported_now.get(k)),
        "retagged": retagged,
        "metadata_updated": metadata_updated,
        "metadata_skipped": metadata_skipped,
        "needs_review": sorted(k for k in keys if manifest.entry(k).get("status") == "needs_review"),
        "failed": sorted(k for k in keys if manifest.entry(k).get("status") == "failed"),
        "manifest": str(manifest.path) if manifest.persist else None,
        "cancelled": bool(cancel and cancel()),
        "time_s": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- settings persistence
PERSISTED = ("margin_mm", "interpolation", "tiff_compression", "overlay", "ppi", "profile", "profiles_file", "strict",
             "flatten", "shadows")


def settings_to_dict(s: Settings) -> dict:
    d = {k: getattr(s, k) for k in PERSISTED}
    d["derivatives"] = [asdict(x) for x in s.derivatives]
    d["rules"] = asdict(s.rules)
    if s.photo_crop != "off":      # defaults are left out: a manifest made without the photo crop stays as it was
        d["photo_crop"] = s.photo_crop
    if s.photo_bleed_mm != PHOTO_BLEED_MM:
        d["photo_bleed_mm"] = s.photo_bleed_mm
    if s.metadata:      # left out when empty: a manifest made without metadata stays as it was
        d["metadata"] = s.metadata
    return d


def settings_from_dict(d: dict) -> Settings:
    s = Settings()
    for k in PERSISTED:
        if k in d:
            setattr(s, k, d[k])
    for k in ("photo_crop", "photo_bleed_mm"):
        if k in d:
            setattr(s, k, d[k])
    if d.get("metadata"):      # tolerant: a hand-edited manifest keeps its texts literally, never raises
        s.metadata = meta_mod.normalize_stored(d["metadata"])
    if "derivatives" in d:
        s.derivatives = tuple(Derivative(**x) for x in d["derivatives"])
    if "rules" in d:
        s.rules = ReviewRules(**d["rules"])
    return s
