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
import json
import sys
from pathlib import Path

COMMANDS = ("eda", "overlays", "compare-overlay", "stages", "pitch", "workload")

# Insert src/ onto the path so we can import soccertrack
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualisation and Phase 3 workload analytics")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None)
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="win60s")
    parser.add_argument("--pipeline", default=None, help="<detector>+<tracker>")
    parser.add_argument("--use-fixtures", action="store_true", help="render from tests/fixtures/ (no real data)")
    parser.add_argument('--output-dir',type=Path,help='Fresh contained figure root (default outputs/figures)')
    parser.add_argument('--left-pipeline',default=None)
    parser.add_argument('--right-pipeline',default='yolov8n_coco+bytetrack')
    return parser

def _figure_root(args):
    root=(args.output_dir or _REPO/'outputs/figures').resolve()
    if not root.is_relative_to(_REPO/'outputs'):raise ValueError('Figures must remain below repository outputs')
    return root


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _cmd_eda(args, cfg) -> None:
    """Generate EDA figures from outputs/eda/stats.json (or fixtures)."""
    import pandas as pd
    from soccertrack.visualize import plot_eda

    fixtures_dir = _REPO / "tests" / "fixtures"

    if args.use_fixtures:
        stats_path = fixtures_dir / "eda_stats_sample.json"
        box_sample_path = None
        print(f"[EDA] Using fixture: {stats_path}")
    else:
        stats_path = _REPO / "outputs" / "eda" / "stats.json"
        box_sample_path = _REPO / "outputs" / "eda" / "box_sample.csv"
        if not stats_path.exists():
            print(
                f"[EDA] stats.json not found at {stats_path}\n"
                "      Run with --use-fixtures to render from fixture data, or\n"
                "      wait for Member 1 to produce outputs/eda/stats.json.",
                file=sys.stderr,
            )
            sys.exit(1)

    with stats_path.open("r", encoding="utf-8") as fh:
        stats = json.load(fh)

    box_sample = None
    if box_sample_path and box_sample_path.exists():
        box_sample = pd.read_csv(box_sample_path)

    out_dir = _figure_root(args) / 'eda'
    produced = plot_eda(stats, box_sample, out_dir)
    print(f"[EDA] Wrote {len(produced)} figures to {out_dir}:")
    for p in produced:
        print(f"  {p.name}")


def _cmd_overlays(args, cfg, sources) -> None:
    """Draw bounding-box overlays for one pipeline on the requested match/frameset."""
    from soccertrack import config, contracts
    from soccertrack.visualize import draw_boxes

    if args.pipeline is None:
        print("[overlays] --pipeline is required (e.g. yolov8n_coco+bytetrack)", file=sys.stderr)
        sys.exit(1)

    detector, _, tracker = args.pipeline.partition("+")
    kind = "tracks" if tracker else "detections"

    # Resolve match IDs
    split_cfg = cfg.get("project", {}).get("split", {})
    match_ids: list[str] = list(split_cfg.get(args.split, []))
    if args.match:
        match_ids = [args.match]

    for mid in match_ids:
        table_path = _REPO / "outputs" / kind / args.pipeline / args.split / f"{mid}__{args.frameset}.csv"
        if not table_path.exists():
            print(f"[overlays] Table not found (yet): {table_path}")
            continue
        table = contracts.read_table(table_path, kind)
        meta=contracts.read_run_meta(table_path)
        f=meta['frames'];frames=range(f['start'],f['stop'],f['stride'])

        out_dir = _figure_root(args) / 'overlays'
        out_dir.mkdir(parents=True, exist_ok=True)

        # We need real frames from data_io.iter_frames; that depends on Member 1
        try:
            from soccertrack.data_io import iter_frames  # type: ignore
            mp = config.match_paths(mid, sources)
            for frame_id, bgr in iter_frames(mp.clip, list(frames)[:5]):  # first 5 frames as sample
                img = draw_boxes(bgr, table, frame_id)
                _write_img(img, out_dir / f"{mid}_{args.pipeline}_{frame_id:06d}.png")
            print(f"[overlays] Wrote overlay samples for match {mid} to {out_dir}")
        except (ImportError, NotImplementedError):
            print(f"[overlays] data_io.iter_frames not available yet — skipping frame decode for {mid}")


def _cmd_compare_overlay(args, cfg, sources) -> None:
    """Side-by-side (vertically stacked) overlay for Pipeline A vs Pipeline B."""
    from soccertrack import config, contracts
    from soccertrack.visualize import side_by_side

    pipeline_a = args.left_pipeline
    if pipeline_a is None:
        latest=_REPO/'outputs/phase2_final/latest.json'
        if latest.exists():
            accepted=json.loads((_REPO/json.loads(latest.read_text())['path']/'accepted_results.json').read_text())
            pipeline_a=accepted['selected_scratch_detector']+'+iou_hungarian'
        else:pipeline_a='scratch_hog_svm+iou_hungarian'
    pipeline_b = args.right_pipeline
    contracts.split_pipeline_id(pipeline_a);contracts.split_pipeline_id(pipeline_b)

    split_cfg = cfg.get("project", {}).get("split", {})
    match_ids = list(split_cfg.get(args.split, []))
    if args.match:
        match_ids = [args.match]

    for mid in match_ids:
        path_a = _REPO / "outputs" / "tracks" / pipeline_a / args.split / f"{mid}__{args.frameset}.csv"
        path_b = _REPO / "outputs" / "tracks" / pipeline_b / args.split / f"{mid}__{args.frameset}.csv"

        if not path_a.exists() or not path_b.exists():
            raise FileNotFoundError(f'Comparison tracks missing for {mid}; specify accepted pipelines')

        table_a = contracts.read_table(path_a, "tracks")
        table_b = contracts.read_table(path_b, "tracks")

        try:
            from soccertrack.data_io import iter_frames  # type: ignore
            mp = config.match_paths(mid, sources)
            out_dir = _figure_root(args) / 'overlays'
            out_dir.mkdir(parents=True, exist_ok=True)
            for frame_id, bgr in iter_frames(mp.clip, [0]):  # first frame as sample
                img = side_by_side(bgr, table_a, table_b, frame_id,
                                   titles=("Pipeline A: Scratch+IoU", "Pipeline B: YOLOv8n+ByteTrack"))
                _write_img(img, out_dir / f"{mid}_compare_{frame_id:06d}.png")
            print(f"[compare-overlay] Wrote comparison for match {mid}")
        except (ImportError, NotImplementedError):
            print("[compare-overlay] data_io.iter_frames not available yet — skipping frame decode")


def _cmd_stages(args, cfg) -> None:
    """Compose a segmentation stage figure from Member 1's stage images."""
    from soccertrack.visualize import plot_stage_figure

    split_cfg = cfg.get("project", {}).get("split", {})
    match_ids = list(split_cfg.get(args.split, []))
    if args.match:
        match_ids = [args.match]

    for mid in match_ids:
        stage_dir = _REPO / "outputs" / "stages" / mid
        if not stage_dir.exists():
            print(f"[stages] Stage dir not found (yet): {stage_dir}")
            continue
        out_dir = _figure_root(args) / 'overlays'
        out_path = out_dir / f"{mid}_stage.png"
        plot_stage_figure(stage_dir, frame_id=0, out_path=out_path)
        print(f"[stages] Wrote stage figure: {out_path}")


def _write_img(img, path: Path) -> None:
    """Write a BGR numpy array as PNG."""
    import cv2  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    from soccertrack import config
    from soccertrack import split
    if args.split=='test': raise PermissionError('Held-out visualization prohibited')

    try:
        cfg = config.load_config()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    if args.match is not None and split.split_of(args.match,cfg)!=args.split:
        raise PermissionError('Match/split mismatch')

    # data_sources only needed for commands that decode real video
    sources: dict = {}
    _needs_sources = args.command in ("overlays", "compare-overlay")
    if _needs_sources and not args.use_fixtures:
        try:
            sources = config.load_data_sources()
        except (FileNotFoundError, ValueError) as exc:
            print(f"[WARNING] {exc}\nFalling back to fixture-only mode.", file=sys.stderr)

    if args.command == "eda":
        _cmd_eda(args, cfg)
    elif args.command == "overlays":
        _cmd_overlays(args, cfg, sources)
    elif args.command == "compare-overlay":
        _cmd_compare_overlay(args, cfg, sources)
    elif args.command == "stages":
        _cmd_stages(args, cfg)
    elif args.command in ("pitch", "workload"):
        print(
            f"[{args.command}] Phase 3 command — not yet implemented (Phase 3 milestone).",
            file=sys.stderr,
        )
        sys.exit(0)
    else:
        print(f"[ERROR] Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
