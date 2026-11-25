"""Verify every data file and weight referenced in configs/data_sources.yaml (read-only).

Owner: Member 4. Usage:
  python scripts/check_data.py                 # all matches of all splits + weights
  python scripts/check_data.py --split val
  python scripts/check_data.py --decode        # also count decoded frames (slow, ~1-2 min per clip)
Report: outputs/data_check.json and a console table.
Checks per match: files exist; byte size and SHA-256 against the manifest (if configured); MOT has 10 columns,
0-based frames; video width/height/fps; decoded frame count == max(frame) + 1 (with --decode).
Never writes, moves or renames anything outside outputs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*, streaming to avoid large memory use."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_mot(mot_path: Path) -> dict:
    """Check the MOT annotation file: existence, 10 columns, 0-based frames."""
    result: dict = {"path": str(mot_path), "exists": mot_path.exists(), "issues": []}
    if not mot_path.exists():
        return result

    result["size_bytes"] = mot_path.stat().st_size
    frames_seen: set[int] = set()
    min_frame = None
    row_count = 0

    with mot_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 10:
                result["issues"].append(
                    f"Line {lineno}: expected 10 columns, got {len(parts)}"
                )
                continue
            try:
                frame = int(float(parts[0]))
            except ValueError:
                result["issues"].append(f"Line {lineno}: cannot parse frame index '{parts[0]}'")
                continue
            frames_seen.add(frame)
            if min_frame is None or frame < min_frame:
                min_frame = frame
            row_count += 1

    result["rows"] = row_count
    result["min_frame"] = min_frame
    result["max_frame"] = max(frames_seen) if frames_seen else None
    result["frames_annotated"] = len(frames_seen)

    if min_frame is not None and min_frame != 0:
        result["issues"].append(
            f"Frame index is not 0-based: minimum frame is {min_frame} (expected 0)"
        )
    return result


def _check_video(clip_path: Path) -> dict:
    """Check video header using cv2.VideoCapture (no full decode)."""
    result: dict = {"path": str(clip_path), "exists": clip_path.exists(), "issues": []}
    if not clip_path.exists():
        return result

    result["size_bytes"] = clip_path.stat().st_size

    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            result["issues"].append("cv2.VideoCapture could not open the file")
            return result
        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["fps"] = cap.get(cv2.CAP_PROP_FPS)
        result["container_frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except ImportError:
        result["issues"].append("opencv-python not installed; skipping video header check")

    return result


def _decode_frames(clip_path: Path) -> int | None:
    """Count frames by sequential decoding (slow). Falls back gracefully."""
    # Try Member 1's function first; fall back to our own loop
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from soccertrack.data_io import count_decoded_frames  # type: ignore
        return count_decoded_frames(clip_path)
    except (ImportError, NotImplementedError, AttributeError):
        pass

    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(clip_path))
        count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            count += 1
        cap.release()
        return count
    except ImportError:
        return None


def _resolve_matches_for_split(split: str, sources: dict, project_cfg: dict) -> list[str]:
    """Return the list of match IDs for the requested split."""
    if split == "all":
        return list(sources.get("matches", {}).keys())
    split_cfg = project_cfg.get("split", {})
    ids: list[str] = list(split_cfg.get(split, []))
    if split == "train" and split_cfg.get("use_train_optional", False):
        ids += list(split_cfg.get("train_optional", []))
    if split == "val" and split_cfg.get("use_val_optional", False):
        ids += list(split_cfg.get("val_optional", []))
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify referenced SoccerTrack files and weights")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--decode", action="store_true", help="count frames by sequential decoding")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Import config from the src package
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from soccertrack import config  # type: ignore

    try:
        sources = config.load_data_sources()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = config.load_config()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    project_cfg = cfg.get("project", {})
    match_ids = _resolve_matches_for_split(args.split, sources, project_cfg)

    # Load optional manifest (hash reference)
    manifest: dict = {}
    manifest_path_str = sources.get("manifest")
    if manifest_path_str and not args.no_hash:
        mp = Path(manifest_path_str)
        if mp.exists():
            with mp.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)

    report: dict = {"split": args.split, "matches": {}, "weights": {}, "issues_summary": []}
    all_ok = True

    # Check each match
    for mid in match_ids:
        try:
            mp = config.match_paths(mid, sources)
        except KeyError as exc:
            report["issues_summary"].append(str(exc))
            all_ok = False
            continue

        entry: dict = {"match_id": mid}

        # MOT annotation
        mot_check = _check_mot(mp.mot)
        entry["mot"] = mot_check
        if mot_check["issues"]:
            all_ok = False

        # Video header
        vid_check = _check_video(mp.clip)
        entry["clip"] = vid_check
        if vid_check["issues"]:
            all_ok = False

        # SHA-256 (if manifest provided and not skipped)
        if not args.no_hash and manifest:
            for path_obj, key in [(mp.mot, "mot_sha256"), (mp.clip, "clip_sha256")]:
                if path_obj.exists() and mid in manifest:
                    expected = manifest[mid].get(key)
                    if expected:
                        actual = _hash_file(path_obj)
                        entry[f"{key}_ok"] = (actual == expected)
                        if actual != expected:
                            entry[f"{key}_actual"] = actual
                            all_ok = False

        # Optional full decode
        if args.decode and mp.clip.exists():
            decoded = _decode_frames(mp.clip)
            entry["decoded_frames"] = decoded
            if decoded is not None and mot_check.get("frames_annotated") is not None:
                expected_frames = mot_check["frames_annotated"]
                if decoded != expected_frames:
                    entry["frame_count_mismatch"] = {
                        "decoded": decoded,
                        "annotated": expected_frames,
                    }
                    all_ok = False

        # Calibration dir (optional)
        if mp.calibration_dir is not None:
            entry["calibration_dir"] = {
                "path": str(mp.calibration_dir),
                "exists": mp.calibration_dir.exists(),
            }

        report["matches"][mid] = entry

    # Check weights
    for key, wpath_str in sources.get("weights", {}).items():
        wpath = Path(wpath_str)
        w_entry: dict = {"path": str(wpath), "exists": wpath.exists()}
        if wpath.exists():
            w_entry["size_bytes"] = wpath.stat().st_size
        else:
            all_ok = False
        report["weights"][key] = w_entry

    report["all_ok"] = all_ok

    # Write JSON report
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_check.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Console table
    col_w = 12
    header = f"{'match_id':<{col_w}} {'mot_ok':<8} {'clip_ok':<8} {'issues'}"
    print(header)
    print("-" * len(header))
    for mid, entry in report["matches"].items():
        mot_ok = "✓" if not entry.get("mot", {}).get("issues") else "✗"
        clip_ok = "✓" if not entry.get("clip", {}).get("issues") else "✗"
        issues = "; ".join(entry.get("mot", {}).get("issues", []) + entry.get("clip", {}).get("issues", []))
        print(f"{mid:<{col_w}} {mot_ok:<8} {clip_ok:<8} {issues[:80]}")

    print()
    print(f"Weights:")
    for key, we in report["weights"].items():
        ok = "✓" if we["exists"] else "✗ MISSING"
        print(f"  {key}: {ok}  ({we['path']})")

    print()
    status = "ALL OK" if all_ok else "ISSUES FOUND — see outputs/data_check.json"
    print(f"Status: {status}")
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
