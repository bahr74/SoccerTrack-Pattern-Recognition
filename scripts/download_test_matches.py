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
    # TODO(Member 4, M0): huggingface_hub.hf_hub_download per explicit filename (never snapshot the whole repo:
    # it is ~145 GB), repo_type="dataset", revision=REVISION, local_dir=<repo>/data/raw/soccertrack-v2.
    # Refuse to run if HF_HOME is unset or on C:. Never print or store the token.
    raise NotImplementedError("TODO(Member 4, M0)")


if __name__ == "__main__":
    main()
