"""Member 3 entry point: evaluations E1-E4, oracle inputs and comparison tables.

Owner: Member 3. Usage (PowerShell, from the repository root, with .venv active):
  python scripts/run_evaluation.py oracle    --split val --frameset win60s      # writes gt_oracle and gt_noisy detections
  python scripts/run_evaluation.py patches   --detector scratch_hog_svm --split val                     # E1
  python scripts/run_evaluation.py proposals --detector scratch_hog_svm --split val --frameset eval10   # E2
  python scripts/run_evaluation.py detection --detector yolov8n_coco --split val --frameset eval10      # E3
  python scripts/run_evaluation.py tracking  --pipeline yolov8n_coco+bytetrack --split val --frameset win60s  # E4
  python scripts/run_evaluation.py compare   --split val
  python scripts/run_evaluation.py fixtures                                     # evaluate tests/fixtures (no data)
Test split: add --frozen-test (only during the logged frozen test run).
"""

from __future__ import annotations

import argparse

COMMANDS = ("oracle", "patches", "proposals", "detection", "tracking", "compare", "fixtures", "failures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluation E1-E4 and model comparison")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None, help="one match ID; default = every match of --split (+ pooled)")
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="eval10")
    parser.add_argument("--detector", default=None)
    parser.add_argument("--pipeline", default=None, help="<detector>+<tracker>")
    parser.add_argument("--frozen-test", action="store_true", help="required for --split test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # TODO(Member 3): dispatch to evaluate / compare / explain ("failures" is Phase 3, train/val only).
    raise NotImplementedError(f"TODO(Member 3): command '{args.command}'")


if __name__ == "__main__":
    main()
