"""Match-level split, leakage guards and the test-access log.

Owner: Member 1. Priority milestone M1 (day 1): runners of Members 1 and 2 call `require_test_access`.
Contract: SHARED_CONTRACTS.md Section 14; plan Section 5.2.
The split values live in configs/project.yaml and are frozen by the approved plan.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from soccertrack import config

_SPLITS = ("train", "val", "test")


def _project(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept load_config()'s bundle or the project.yaml mapping used in tests."""
    return cfg.get("project", cfg)


def _check_split(split: str) -> None:
    if split not in _SPLITS:
        raise ValueError(f"Unknown split: {split!r}; expected train, val or test")


def matches_for(split: str, cfg: dict[str, Any]) -> list[str]:
    """Return the match IDs of `split` ("train" | "val" | "test"), honouring the use_*_optional flags."""
    _check_split(split)
    assert_disjoint(cfg)
    splits = _project(cfg)["split"]
    matches = list(splits[split])
    if splits.get(f"use_{split}_optional", False):
        matches.extend(splits.get(f"{split}_optional", []))
    return matches


def split_of(match_id: str, cfg: dict[str, Any]) -> str:
    """Return the split a match belongs to; raise for unknown matches."""
    assert_disjoint(cfg)
    for split in _SPLITS:
        if match_id in matches_for(split, cfg):
            return split
    raise ValueError(f"Unknown or disabled match: {match_id!r}")


def assert_disjoint(cfg: dict[str, Any]) -> None:
    """Raise if any match appears in more than one split (including optional matches)."""
    splits = _project(cfg)["split"]
    seen: dict[str, str] = {}
    for split in _SPLITS:
        for match in list(splits[split]) + list(splits.get(f"{split}_optional", [])):
            if not isinstance(match, str) or not match.isascii() or not match.isdigit():
                raise ValueError("Match IDs must be strings of ASCII digits")
            if match in seen:
                raise ValueError(f"Match {match} repeated in {seen[match]} and {split}")
            seen[match] = split


def require_test_access(split: str, frozen_test: bool) -> None:
    """Raise PermissionError when `split == "test"` and the caller did not pass --frozen-test."""
    _check_split(split)
    if split == "test" and frozen_test is not True:
        raise PermissionError("Test split access requires --frozen-test")


def log_test_access(member: str, method: str, artefacts: list[str], model_hashes: dict[str, str]) -> None:
    """Append one entry (UTC date, member, method, config_sha256, model hashes, artefacts) to outputs/test_log.md."""
    project = _project(config.load_config())
    root = config.repo_root().resolve()
    normalized = [_artifact_relative(path, root) for path in artefacts]
    log = (root / Path(project["paths"]["test_log"])).resolve()
    outputs = (root / Path(project["paths"]["outputs_root"])).resolve()
    if not outputs.is_relative_to(root) or outputs == root or not log.is_relative_to(outputs):
        raise ValueError("Test log must be inside the repository outputs directory")
    entry = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "member": member,
        "method": method,
        "config_sha256": config.config_sha256(),
        "model_hashes": model_hashes,
        "artefacts": normalized,
    }
    text = "\n```json\n" + json.dumps(entry, indent=2, ensure_ascii=False) + "\n```\n"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _artifact_relative(value: str, root: Path) -> str:
    """Validate Windows and POSIX paths on either platform before any log write."""
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError("Artifact paths must be nonempty repository-relative paths")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.drive or windows.root or posix.is_absolute() or ":" in value:
        raise ValueError("Artifact paths must not be absolute, drive-qualified or UNC")
    parts: list[str] = []
    for part in posix.parts:
        if part == "..":
            if not parts:
                raise ValueError("Artifact path escapes the repository")
            parts.pop()
        elif part != ".":
            parts.append(part)
    if not parts or not root.joinpath(*parts).resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes the repository or names its root")
    return PurePosixPath(*parts).as_posix()
