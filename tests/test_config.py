"""Tests owned by Member 4. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helper: locate repo root (same logic as config.repo_root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# test_01 — load_config: four sections present and split lists are disjoint
# ---------------------------------------------------------------------------
def test_01():
    """load_config returns all four sections; train / val / test sets are disjoint."""
    from soccertrack.config import load_config

    cfg = load_config()

    # All four sections must be present
    for section in ("project", "scratch", "yolo", "evaluation"):
        assert section in cfg, f"Missing section: {section}"

    # Split lists must be disjoint
    project = cfg["project"]
    split = project.get("split", {})
    train = set(split.get("train", []))
    val = set(split.get("val", []))
    test = set(split.get("test", []))

    assert train.isdisjoint(val), f"train ∩ val not empty: {train & val}"
    assert train.isdisjoint(test), f"train ∩ test not empty: {train & test}"
    assert val.isdisjoint(test), f"val ∩ test not empty: {val & test}"

    # Each list must be non-empty
    assert train, "train split is empty"
    assert val, "val split is empty"
    assert test, "test split is empty"


# ---------------------------------------------------------------------------
# test_02 — load_data_sources: rejects unfilled placeholders
# ---------------------------------------------------------------------------
def test_02(tmp_path):
    """load_data_sources raises ValueError when <ST_LOCAL> placeholder remains."""
    from soccertrack.config import load_data_sources

    # Populate a fake data_sources.yaml that still has a placeholder
    bad_yaml = tmp_path / "data_sources.yaml"
    bad_yaml.write_text(
        textwrap.dedent("""\
            matches:
              "117093": {mot: <ST_LOCAL>/mot/117093.txt, clip: <ST_LOCAL>/clip/117093.mp4, calibration_dir: null}
            read_only_roots:
              - <ST_LOCAL>
            weights:
              yolov8n_coco: <ST_LOCAL>/models/yolov8n.pt
            """),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="<ST_LOCAL>"):
        load_data_sources(path=bad_yaml)


def test_02b(tmp_path):
    """load_data_sources raises FileNotFoundError when the file is missing."""
    from soccertrack.config import load_data_sources

    missing = tmp_path / "data_sources.yaml"
    with pytest.raises(FileNotFoundError):
        load_data_sources(path=missing)


# ---------------------------------------------------------------------------
# test_03 — artifact_path: correct path structure; refuses read-only roots
# ---------------------------------------------------------------------------
def test_03():
    """artifact_path builds outputs/detections/<det>/<split>/<match>__<frameset>.csv."""
    from soccertrack.config import artifact_path

    p = artifact_path("detections", "yolov8n_coco", "val", "117093", "eval10")

    # Must be absolute and inside <repo>/outputs/
    assert p.is_absolute()
    assert p.suffix == ".csv"
    assert "outputs" in p.parts
    assert "detections" in p.parts
    assert "yolov8n_coco" in p.parts
    assert "val" in p.parts
    assert p.stem == "117093__eval10"

    # Parent directory must have been created
    assert p.parent.exists()


def test_03_readonly_guard(tmp_path):
    """artifact_path refuses to write inside a read_only_roots entry."""
    from soccertrack.config import artifact_path

    # Build a sources dict whose read_only_roots contains the repo's outputs dir
    outputs_dir = REPO / "outputs"
    sources = {"read_only_roots": [str(outputs_dir)]}

    with pytest.raises(PermissionError):
        artifact_path("detections", "yolov8n_coco", "val", "117093", "eval10", sources=sources)


# ---------------------------------------------------------------------------
# test_04 — frameset_range: correct range values and clamping
# ---------------------------------------------------------------------------
def test_04():
    """frameset_range('eval10', 5976) == range(0, 5976, 10); win60s is clamped."""
    from soccertrack.config import load_config, frameset_range

    cfg = load_config()

    # eval10: stride 10, stop=null → num_frames
    r = frameset_range("eval10", 5976, cfg)
    assert r == range(0, 5976, 10), f"eval10 range wrong: {r}"

    # win60s: stop=1500, stride=1 — not clamped when clip is longer
    r2 = frameset_range("win60s", 6000, cfg)
    assert r2 == range(0, 1500, 1), f"win60s range wrong: {r2}"

    # win60s clamped when clip is shorter than 1500 frames
    r3 = frameset_range("win60s", 100, cfg)
    assert r3 == range(0, 100, 1), f"win60s not clamped: {r3}"

    # fixture3: start=0, stop=3, stride=1
    r4 = frameset_range("fixture3", 999, cfg)
    assert r4 == range(0, 3, 1), f"fixture3 range wrong: {r4}"


# ---------------------------------------------------------------------------
# test_05 — config_sha256: stable across line-ending changes
# ---------------------------------------------------------------------------
def test_05(tmp_path, monkeypatch):
    """config_sha256 is stable across \\r\\n / \\n line endings; ignores data_sources.yaml."""
    # We test stability by computing the hash twice and checking they match,
    # and by verifying that adding/removing data_sources.yaml has no effect.

    from soccertrack.config import config_sha256

    h1 = config_sha256()
    assert isinstance(h1, str)
    assert len(h1) == 64, "SHA-256 digest must be 64 hex characters"

    h2 = config_sha256()
    assert h1 == h2, "config_sha256 must be deterministic"

    # Verify it covers exactly the four committed config files, not data_sources.yaml
    # We recompute manually with normalised line endings
    manual = hashlib.sha256()
    for fname in ("project.yaml", "scratch.yaml", "yolo.yaml", "evaluation.yaml"):
        raw = (REPO / "configs" / fname).read_bytes()
        manual.update(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
    assert h1 == manual.hexdigest(), "config_sha256 disagrees with manual SHA-256 computation"


# ---------------------------------------------------------------------------
# data-marked test (requires real data_sources.yaml + data files)
# ---------------------------------------------------------------------------
@pytest.mark.data
def test_data_01():
    """Every train/val path and weights path in configs/data_sources.yaml exists (read-only check)."""
    from soccertrack.config import load_data_sources, load_config, match_paths

    sources = load_data_sources()
    cfg = load_config()
    project = cfg["project"]
    split = project.get("split", {})

    checked = []
    missing = []

    for split_name in ("train", "val"):
        for mid in split.get(split_name, []):
            try:
                mp = match_paths(str(mid), sources)
                for label, path in [("mot", mp.mot), ("clip", mp.clip)]:
                    checked.append(path)
                    if not path.exists():
                        missing.append(f"{mid}/{label}: {path}")
            except KeyError as exc:
                missing.append(str(exc))

    for key, wpath in sources.get("weights", {}).items():
        p = Path(wpath)
        checked.append(p)
        if not p.exists():
            missing.append(f"weights/{key}: {p}")

    assert not missing, (
        f"{len(missing)} path(s) missing out of {len(checked)} checked:\n"
        + "\n".join(f"  {m}" for m in missing)
    )


# ---------------------------------------------------------------------------
# test_06 — scripts/check_data.py MOT and video helpers on tmp_path
# ---------------------------------------------------------------------------
def test_06_check_data_helpers(tmp_path):
    """check_data._check_mot validates 10-column 0-based MOT files and flags 1-based ones."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_data", REPO / "scripts" / "check_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    good_mot = tmp_path / "good.txt"
    good_mot.write_text("0,1,10,20,30,40,-1,-1,-1,-1\n1,1,11,20,30,40,-1,-1,-1,-1\n", encoding="utf-8")
    res_good = mod._check_mot(good_mot)
    assert res_good["exists"] is True
    assert res_good["rows"] == 2
    assert res_good["min_frame"] == 0
    assert res_good["issues"] == []

    bad_mot = tmp_path / "bad.txt"
    bad_mot.write_text("1,1,10,20,30,40,-1,-1,-1,-1\n", encoding="utf-8")
    res_bad = mod._check_mot(bad_mot)
    assert any("0-based" in iss for iss in res_bad["issues"])


# ---------------------------------------------------------------------------
# test_07 — scripts/download_test_matches.py --dry-run (no network)
# ---------------------------------------------------------------------------
def test_07_download_dry_run(capsys, monkeypatch):
    """download_test_matches.py --dry-run lists test match files without network access."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "download_test_matches", REPO / "scripts" / "download_test_matches.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setattr(sys, "argv", ["download_test_matches.py", "--dry-run"])
    mod.main()
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "128057" in out
    assert "132831" in out
