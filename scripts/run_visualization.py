"""Member 4 entry point: EDA figures, overlays, stage figures and (Phase 3) pitch/workload figures.

Owner: Member 4. Usage (PowerShell, from the repository root, with .venv active):
  python scripts/run_visualization.py eda                         # from outputs/eda/ (or --use-fixtures)
  python scripts/run_visualization.py overlays --pipeline yolov8n_coco+bytetrack --split val --frameset win60s
  python scripts/run_visualization.py compare-overlay --split val --frameset win60s   # Pipeline A vs B
  python scripts/run_visualization.py stages --match 117093
  python scripts/run_visualization.py pitch    --pipeline yolov8n_ft+bytetrack --split val   # Phase 3
  python scripts/run_visualization.py workload --pipeline yolov8n_ft+bytetrack --split val   # Phase 3
"""

from __future__ import annotations

import argparse

COMMANDS = ("eda", "overlays", "compare-overlay", "stages", "pitch", "workload")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation and Phase 3 workload analytics")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None)
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="win60s")
    parser.add_argument("--pipeline", default=None, help="<detector>+<tracker>")
    parser.add_argument("--use-fixtures", action="store_true", help="render from tests/fixtures/ (no real data)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # TODO(Member 4): dispatch to visualize / pitch / workload. Visualisation never changes evaluation results.
    raise NotImplementedError(f"TODO(Member 4): command '{args.command}'")


if __name__ == "__main__":
    main()
