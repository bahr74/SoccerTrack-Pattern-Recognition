"""Configuration loading, path resolution and run provenance.

Owner: Member 4. Priority milestone M0 (first half-day): every other member imports this module.
Contract: docs/team/SHARED_CONTRACTS.md, Sections 2 and 13.

Rules:
- Read configs/project.yaml, scratch.yaml, yolo.yaml and evaluation.yaml from the repository.
- Read machine paths ONLY from configs/data_sources.yaml (git-ignored).
- Never create, modify or delete anything below a `read_only_roots` entry.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = ("project.yaml", "scratch.yaml", "yolo.yaml", "evaluation.yaml")
DATA_SOURCES_FILE = "data_sources.yaml"

# Placeholders that must not appear in a filled-in data_sources.yaml
_PLACEHOLDERS = ("<ST_LOCAL>", "<REPO>")


@dataclass(frozen=True)
class MatchPaths:
    """Absolute paths for one match, taken from configs/data_sources.yaml."""

    match_id: str
    mot: Path
    clip: Path
    calibration_dir: Path | None


# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    """Return the repository root (the folder that contains ``configs/``).

    Resolved from this file's location; independent of the current working
    directory.  ``src/soccertrack/config.py``  →  ``../../``
    """
    # This file is at <repo>/src/soccertrack/config.py
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict[str, Any]:
    """Load and return ``{"project": …, "scratch": …, "yolo": …, "evaluation": …}``.

    Raises ``FileNotFoundError`` with a clear message if any committed config
    file is missing.
    """
    root = repo_root()
    cfg: dict[str, Any] = {}
    for filename in CONFIG_FILES:
        section = filename.replace(".yaml", "")
        config_path = root / "configs" / filename
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Expected one of the four committed config files in {root / 'configs'}."
            )
        with config_path.open("r", encoding="utf-8") as fh:
            cfg[section] = yaml.safe_load(fh) or {}
    return cfg


def load_data_sources(path: Path | None = None) -> dict[str, Any]:
    """Load ``configs/data_sources.yaml`` (or *path*) and validate its structure.

    Raises a helpful ``FileNotFoundError`` pointing to
    ``configs/data_sources.example.yaml`` when the file is missing, and a
    ``ValueError`` when any ``<ST_LOCAL>`` / ``<REPO>`` placeholder remains.
    """
    if path is None:
        path = repo_root() / "configs" / DATA_SOURCES_FILE

    example = repo_root() / "configs" / "data_sources.example.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Machine-specific data sources file not found: {path}\n"
            f"Copy the template and fill in your paths:\n"
            f"  copy {example} {path}\n"
            f"Then replace every <ST_LOCAL> and <REPO> with absolute paths on your machine."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw_text = fh.read()

    for placeholder in _PLACEHOLDERS:
        if placeholder in raw_text:
            raise ValueError(
                f"configs/data_sources.yaml still contains the placeholder '{placeholder}'.\n"
                f"Open the file and replace every <ST_LOCAL> and <REPO> with absolute paths.\n"
                f"See {example} for the expected structure."
            )

    return yaml.safe_load(raw_text) or {}


def match_paths(match_id: str, sources: dict[str, Any]) -> MatchPaths:
    """Return the :class:`MatchPaths` for *match_id* (a string such as ``"117093"``).

    Raises ``KeyError`` with a helpful message if the match is not listed in
    ``data_sources.yaml``.
    """
    matches = sources.get("matches", {})
    if match_id not in matches:
        available = ", ".join(str(k) for k in matches)
        raise KeyError(
            f"Match '{match_id}' not found in configs/data_sources.yaml.\n"
            f"Available matches: {available}"
        )
    entry = matches[match_id]
    cal = entry.get("calibration_dir")
    return MatchPaths(
        match_id=match_id,
        mot=Path(entry["mot"]),
        clip=Path(entry["clip"]),
        calibration_dir=Path(cal) if cal else None,
    )


def weights_path(key: str, sources: dict[str, Any]) -> Path:
    """Return the absolute :class:`~pathlib.Path` of a weights file by *key*
    (e.g. ``"yolov8n_coco"``).

    Raises ``KeyError`` if the key is not in ``data_sources.yaml → weights``.
    """
    weights = sources.get("weights", {})
    if key not in weights:
        available = ", ".join(weights)
        raise KeyError(
            f"Weights key '{key}' not found in configs/data_sources.yaml → weights.\n"
            f"Available keys: {available}"
        )
    return Path(weights[key])


# ---------------------------------------------------------------------------
# Frameset helper
# ---------------------------------------------------------------------------

def frameset_range(name: str, num_frames: int, cfg: dict[str, Any]) -> range:
    """Return the frames of a named frameset from ``configs/evaluation.yaml``.

    ``stop=null`` in the YAML means *end of clip* and is clamped to
    ``num_frames``.  ``stop`` values larger than ``num_frames`` are also
    clamped so callers never receive out-of-bounds indices.
    """
    framesets = cfg.get("evaluation", cfg).get("framesets", {})
    if name not in framesets:
        raise KeyError(
            f"Frameset '{name}' not found in configs/evaluation.yaml → framesets.\n"
            f"Available: {', '.join(framesets)}"
        )
    fs = framesets[name]
    start = int(fs.get("start", 0))
    stop_raw = fs.get("stop", None)
    stop = num_frames if stop_raw is None else min(int(stop_raw), num_frames)
    stride = int(fs.get("stride", 1))
    return range(start, stop, stride)


# ---------------------------------------------------------------------------
# Artefact paths
# ---------------------------------------------------------------------------

def artifact_path(
    kind: str,
    method: str,
    split: str,
    match_id: str,
    frameset: str,
    *,
    sources: dict[str, Any] | None = None,
) -> Path:
    """Return the repository path of a contract artefact
    (SHARED_CONTRACTS.md, Section 2).

    *kind* is one of ``"detections"``, ``"tracks"``, ``"proposals"``,
    ``"pitch"``, ``"workload"``.

    Example::

        artifact_path("detections", "yolov8n_coco", "val", "117093", "eval10")
        # → <repo>/outputs/detections/yolov8n_coco/val/117093__eval10.csv

    The metadata sidecar has the same path with suffix ``.meta.json``.

    Parent directories are created automatically.  The function refuses to
    return a path that falls below any ``read_only_roots`` entry listed in
    ``configs/data_sources.yaml``.
    """
    root = repo_root()
    out_path = root / "outputs" / kind / method / split / f"{match_id}__{frameset}.csv"

    # Enforce read-only-root guard when sources are provided
    if sources is not None:
        resolved_out = out_path.resolve()
        read_only_roots = [Path(r).resolve() for r in sources.get("read_only_roots", [])]
        for ro in read_only_roots:
            try:
                resolved_out.relative_to(ro)
                raise PermissionError(
                    f"artifact_path would write inside a read-only root:\n"
                    f"  path:      {out_path}\n"
                    f"  read-only: {ro}\n"
                    f"Check configs/data_sources.yaml → read_only_roots."
                )
            except ValueError:
                pass  # not relative to this root — fine

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# Config hash (frozen-test provenance)
# ---------------------------------------------------------------------------

def config_sha256() -> str:
    """SHA-256 over the four committed config files, in ``CONFIG_FILES`` order.

    Line endings are normalised to ``\\n`` before hashing so the hash is
    identical on Windows and Linux.  ``configs/data_sources.yaml`` is
    intentionally excluded so frozen-test hashes are machine-independent.
    """
    root = repo_root()
    h = hashlib.sha256()
    for filename in CONFIG_FILES:
        config_path = root / "configs" / filename
        raw = config_path.read_bytes()
        normalised = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        h.update(normalised)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Seed Python ``random``, NumPy and (if installed) PyTorch.

    Enables deterministic cuDNN mode where supported so that results are
    reproducible across runs on the same hardware.
    """
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

