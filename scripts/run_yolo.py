"""Member 2 entry point: Approach B (tiled YOLOv8n) and Pipeline B tracking (ByteTrack).

Owner: Member 2. Usage (PowerShell, from the repository root, with .venv active):
  python scripts/run_yolo.py detect   --detector yolov8n_coco --split val --frameset eval10
  python scripts/run_yolo.py detect   --detector yolov8n_coco --split val --frameset win60s
  python scripts/run_yolo.py track    --detector gt_oracle    --split val --frameset win60s
  python scripts/run_yolo.py track    --detector yolov8n_coco --split val --frameset win60s
  python scripts/run_yolo.py finetune                                   # Phase 3 only
Test split: add --frozen-test (only during the logged frozen test run).
"""

from __future__ import annotations

import argparse

COMMANDS = ("detect", "track", "finetune")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approach B: tiled YOLOv8n detection and ByteTrack tracking")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None, help="one match ID; default = every match of --split")
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="eval10")
    parser.add_argument("--detector", default="yolov8n_coco",
                        help="yolov8n_coco (Phase 3: yolov8n_ft, yolov8s_ft); track also accepts gt_oracle, gt_noisy")
    parser.add_argument("--max-frames", type=int, default=None, help="debug only: process the first N frames")
    parser.add_argument("--frozen-test", action="store_true", help="required for --split test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # TODO(Member 2): load config, call split.require_test_access, dispatch to detector_yolo / tracker_bytetrack,
    # and call split.log_test_access after a successful test-split run. Debug runs (--max-frames) must not be
    # written under outputs/detections/ (use outputs/debug/).
    raise NotImplementedError(f"TODO(Member 2): command '{args.command}'")


if __name__ == "__main__":
    main()
