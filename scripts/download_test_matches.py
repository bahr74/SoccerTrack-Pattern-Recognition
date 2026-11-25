r"""Download ONLY the missing official test matches (128057, 132831) of SoccerTrack v2 release v1.2.

Owner: Member 4. Run by a team member who has accepted the dataset terms on Hugging Face and logged in with
their own token (`huggingface-cli login`). The token is never written to this repository.
Run ONLY after team approval. Total ~0.91 GB (+ ~0.31 GB with --with-val-optional for 132877).

Usage (PowerShell; caches on D: because C: has ~2 GB free):
  $env:HF_HOME = "D:\hf_cache"; $env:TMP = "D:\tmp"; $env:TEMP = "D:\tmp"
  python scripts/download_test_matches.py --dry-run
  python scripts/download_test_matches.py

Files (repo atomscott/soccertrack-v2, revision 325bd4c57ee8e135ab652f3921d274526ada496b):
  mot/README.md, mot/128057.txt, mot/clips/128057.mp4, mot/132831.txt, mot/clips/132831.mp4
  raw/128057/* and raw/132831/* small calibration files (keypoints, intrinsics, homography, padding) only.
Destination: <repo>/data/raw/soccertrack-v2/ (git-ignored). Verify afterwards with scripts/check_data.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HF_REPO = "atomscott/soccertrack-v2"
REVISION = "325bd4c57ee8e135ab652f3921d274526ada496b"
TEST_MATCHES = ("128057", "132831")
OPTIONAL_VAL = ("132877",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the held-out SoccerTrack v2 test matches")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    parser.add_argument("--with-val-optional", action="store_true", help="also download 132877 (needs approval)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # --- Guard: HF_HOME must be set and not on C: for actual downloads ---
    hf_home = os.environ.get("HF_HOME", "")
    if not args.dry_run:
        if not hf_home:
            print(
                "[ERROR] HF_HOME is not set.\n"
                "Set it to a directory on a drive with enough free space, e.g.:\n"
                "  $env:HF_HOME = 'D:\\hf_cache'\n"
                "This keeps the large Hugging Face cache off C:.",
                file=sys.stderr,
            )
            sys.exit(1)
        if Path(hf_home).drive.upper() == "C:":
            print(
                f"[ERROR] HF_HOME is on C: ({hf_home}).\n"
                "C: has limited space. Set HF_HOME to a drive with ≥ 2 GB free, e.g. D:\\hf_cache.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Files to download per match ---
    # Approximate sizes for the dry-run table (MB):
    _FILE_SIZES_MB = {
        "mot/README.md": 0.01,
        "mot/128057.txt": 3.5,
        "mot/clips/128057.mp4": 436.0,
        "mot/132831.txt": 3.5,
        "mot/clips/132831.mp4": 468.0,
        # raw/ calibration files (small)
        "raw/128057/128057_keypoints.json": 0.01,
        "raw/128057/128057_intrinsics.json": 0.01,
        "raw/128057/128057_homography.json": 0.01,
        "raw/128057/128057_padding.json":    0.01,
        "raw/132831/132831_keypoints.json": 0.01,
        "raw/132831/132831_intrinsics.json": 0.01,
        "raw/132831/132831_homography.json": 0.01,
        "raw/132831/132831_padding.json":    0.01,
    }
    # Optional val match 132877
    _OPTIONAL_FILES_MB = {
        "mot/132877.txt": 3.5,
        "mot/clips/132877.mp4": 300.0,
    }

    all_files = dict(_FILE_SIZES_MB)
    if args.with_val_optional:
        all_files.update(_OPTIONAL_FILES_MB)

    local_dir = repo_root / "data" / "raw" / "soccertrack-v2"

    # --- Dry run: list files and sizes ---
    if args.dry_run:
        print(f"DRY RUN — no files will be downloaded.")
        print(f"Repository : {HF_REPO}  (revision {REVISION})")
        print(f"Destination: {local_dir}")
        print()
        total_mb = 0.0
        print(f"{'File':<52} {'~Size (MB)':>12}")
        print("-" * 66)
        for fname, size_mb in all_files.items():
            print(f"  {fname:<50} {size_mb:>12.1f}")
            total_mb += size_mb
        print("-" * 66)
        print(f"  {'TOTAL':<50} {total_mb:>12.1f}")
        print()
        print("To perform the actual download, omit --dry-run.")
        print("You must have accepted the dataset terms on Hugging Face and")
        print("logged in with:  huggingface-cli login")
        return

    # --- Real download ---
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print(
            "[ERROR] huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(1)

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(all_files)} files to {local_dir} …")
    print(f"Repository: {HF_REPO}  revision {REVISION}")
    print()

    failed: list[str] = []
    for fname, size_mb in all_files.items():
        dest = local_dir / fname
        if dest.exists():
            print(f"  [SKIP] already exists: {fname}")
            continue
        print(f"  [GET ] {fname}  (~{size_mb:.0f} MB) …", end="", flush=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                revision=REVISION,
                filename=fname,
                local_dir=str(local_dir),
            )
            print(" done")
        except Exception as exc:  # noqa: BLE001
            print(f" FAILED: {exc}")
            failed.append(fname)

    print()
    if failed:
        print(f"[ERROR] {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  {f}")
        print("Re-run this script to retry the failed files.")
        sys.exit(1)
    else:
        print("All files downloaded successfully.")
        print("Verify with:  python scripts/check_data.py --split test")


if __name__ == "__main__":
    main()
