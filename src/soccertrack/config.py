"""Configuration loading, path resolution and run provenance.

Owner: Member 4. Priority milestone M0 (first half-day): every other member imports this module.
Contract: docs/team/SHARED_CONTRACTS.md, Sections 2 and 13.

Rules:
- Read configs/project.yaml, scratch.yaml, yolo.yaml and evaluation.yaml from the repository.
- Read machine paths ONLY from configs/data_sources.yaml (git-ignored).
- Never create, modify or delete anything below a `read_only_roots` entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_FILES = ("project.yaml", "scratch.yaml", "yolo.yaml", "evaluation.yaml")
DATA_SOURCES_FILE = "data_sources.yaml"


@dataclass(frozen=True)
class MatchPaths:
    """Absolute paths for one match, taken from configs/data_sources.yaml."""

    match_id: str
    mot: Path
    clip: Path
    calibration_dir: Path | None


def repo_root() -> Path:
    """Return the repository root (the folder that contains `configs/`).

    TODO(Member 4, M0): resolve from this file's location; do not depend on the current working directory.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def load_config() -> dict[str, Any]:
    """Load and return {"project": ..., "scratch": ..., "yolo": ..., "evaluation": ...}.

    TODO(Member 4, M0): yaml.safe_load each file in CONFIG_FILES; fail with a clear message if one is missing.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def load_data_sources(path: Path | None = None) -> dict[str, Any]:
    """Load configs/data_sources.yaml (or `path`) and validate its structure.

    TODO(Member 4, M0): raise a helpful error that points to configs/data_sources.example.yaml when the file
    is missing or still contains `<ST_LOCAL>` / `<REPO>` placeholders.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def match_paths(match_id: str, sources: dict[str, Any]) -> MatchPaths:
    """Return the MatchPaths for `match_id` (a string such as "117093").

    TODO(Member 4, M0).
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def weights_path(key: str, sources: dict[str, Any]) -> Path:
    """Return the absolute path of a weights file by key (e.g. "yolov8n_coco").

    TODO(Member 4, M0).
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def frameset_range(name: str, num_frames: int, cfg: dict[str, Any]) -> range:
    """Return the frames of a named frameset from configs/evaluation.yaml (stop=null -> num_frames).

    TODO(Member 4, M0): clamp `stop` to `num_frames`.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def artifact_path(kind: str, method: str, split: str, match_id: str, frameset: str) -> Path:
    """Return the repository path of a contract artefact (SHARED_CONTRACTS.md, Section 2).

    `kind` is one of "detections", "tracks", "proposals", "pitch", "workload".
    Example: artifact_path("detections", "yolov8n_coco", "val", "117093", "eval10")
             -> <repo>/outputs/detections/yolov8n_coco/val/117093__eval10.csv
    The metadata sidecar is the same path with suffix ".meta.json".
    TODO(Member 4, M0): create parent folders; refuse paths below a read-only root.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def config_sha256() -> str:
    """SHA-256 over the four committed config files (not data_sources.yaml), in CONFIG_FILES order.

    TODO(Member 4, M0): normalise line endings to \\n before hashing so Windows and Linux agree.
    """
    raise NotImplementedError("TODO(Member 4, M0)")


def set_seeds(seed: int) -> None:
    """Seed Python `random`, NumPy and (if installed) PyTorch; enable deterministic cuDNN where supported.

    TODO(Member 4, M0).
    """
    raise NotImplementedError("TODO(Member 4, M0)")
