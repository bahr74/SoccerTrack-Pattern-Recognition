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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify referenced SoccerTrack files and weights")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--decode", action="store_true", help="count frames by sequential decoding")
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 (faster)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # TODO(Member 4, M0): implement with config.load_data_sources, hashlib, cv2 (header) and a minimal read of the
    # first MOT column. Use data_io.count_decoded_frames for --decode once Member 1 has delivered it.
    raise NotImplementedError("TODO(Member 4, M0)")


if __name__ == "__main__":
    main()
