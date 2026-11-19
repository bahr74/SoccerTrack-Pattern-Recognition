"""Member 1 entry point: Approach A (from scratch) and Pipeline A tracking.

Owner: Member 1. Usage (PowerShell, from the repository root, with .venv active):
  python scripts/run_scratch.py eda
  python scripts/run_scratch.py segment     --split val
  python scripts/run_scratch.py detect      --detector scratch_mwb0 --split val --frameset eval10
  python scripts/run_scratch.py patches     --split train
  python scripts/run_scratch.py train-svm
  python scripts/run_scratch.py train-cnn
  python scripts/run_scratch.py score-patches --detector scratch_hog_svm --split val
  python scripts/run_scratch.py detect      --detector scratch_hog_svm --split val --frameset eval10
  python scripts/run_scratch.py track       --detector gt_oracle --split val --frameset win60s
  python scripts/run_scratch.py track       --detector scratch_hog_svm --split val --frameset win60s
Test split: add --frozen-test (only during the logged frozen test run).
"""

from __future__ import annotations

import argparse

COMMANDS = ("eda", "segment", "patches", "train-svm", "train-cnn", "score-patches", "detect", "track")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approach A: from-scratch detection and IoU+Hungarian tracking")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None, help="one match ID; default = every match of --split")
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="eval10")
    parser.add_argument("--detector", default="scratch_hog_svm",
                        help="scratch_mwb0 | scratch_hog_svm | scratch_cnn (track also accepts gt_oracle, gt_noisy)")
    parser.add_argument("--frozen-test", action="store_true", help="required for --split test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # TODO(Member 1): load config (config.load_config / load_data_sources), call split.require_test_access,
    # dispatch to data_io / segmentation / patches / classifiers / detector_scratch / tracker_iou,
    # and call split.log_test_access after a successful test-split run.
    raise NotImplementedError(f"TODO(Member 1): command '{args.command}'")


if __name__ == "__main__":
    main()
