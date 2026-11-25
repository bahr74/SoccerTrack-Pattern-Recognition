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

Every command reads producers' files only (never their code) and writes Metrics JSON to outputs/metrics/.
Per-match results are followed by one pooled result when a split has several matches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works with or without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

from soccertrack import contracts, evaluate as E  # noqa: E402

COMMANDS = ("oracle", "patches", "proposals", "detection", "tracking", "compare", "fixtures", "failures")
OUTPUTS = ROOT / "outputs"


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


# ---- Helpers -----------------------------------------------------------------------------------------------------
class SetupMissing(RuntimeError):
    """A shared function owned by another member is not implemented yet."""


def _needs(what: str, fn, *args):
    """Call another member's function; turn NotImplementedError into a clear message."""
    try:
        return fn(*args)
    except NotImplementedError as exc:
        raise SetupMissing(f"{what} is not implemented yet ({exc}). Run this command after that milestone is "
                           "merged, or on the lead machine.") from exc


def artefact(kind: str, method: str, split: str, match: str, frameset: str) -> Path:
    """outputs/<kind>/<method>/<split>/<match>__<frameset>.csv (Contracts §2)."""
    return OUTPUTS / kind / method / split / f"{match}__{frameset}.csv"


def frameset_range(name: str, num_frames: int, cfg: dict) -> range:
    """Frames of a frameset from configs/evaluation.yaml; `stop: null` means the end of the clip."""
    f = cfg["evaluation"]["framesets"][name]
    stop = num_frames if f["stop"] is None else min(int(f["stop"]), num_frames)
    return range(int(f["start"]), stop, int(f["stride"]))


def matches(args, cfg) -> list[str]:
    from soccertrack import split

    split.require_test_access(args.split, args.frozen_test)
    if args.match:
        if split.split_of(args.match, cfg) != args.split:
            raise PermissionError('Match/split mismatch')
        return [args.match]
    return split.matches_for(args.split, cfg)


def ground_truth(match: str):
    from soccertrack import config, data_io

    sources = _needs("config.load_data_sources (Member 4, M0)", config.load_data_sources)
    return _needs("data_io.load_gt_tracks (Member 1) / config.match_paths (Member 4)",
                  data_io.load_gt_tracks, match, sources)


def save(docs: list[dict], pooled: dict | None = None) -> None:
    for doc in docs + ([pooled] if pooled else []):
        path = E.write_metrics(doc, OUTPUTS / "metrics")
        m = doc["metrics"]
        shown = ", ".join(f"{k}={v}" for k, v in m.items() if v is not None)
        print(f"  {doc['run_id']}: {shown}")
        print(f"    -> {path.relative_to(ROOT).as_posix()}")


def log_test(args, method: str, docs: list[dict]) -> None:
    if args.split != "test":
        return
    from soccertrack import split

    paths = [f"outputs/metrics/{d['run_id']}.json" for d in docs]
    _needs("split.log_test_access (needs config, Member 4)", split.log_test_access, "Member 3", method, paths, {})


# ---- Commands ----------------------------------------------------------------------------------------------------
def cmd_oracle(args, cfg) -> None:
    from soccertrack import config, data_io

    selected = matches(args, cfg)  # authorize before even resolving a registry
    sources = _needs("config.load_data_sources (Member 4, M0)", config.load_data_sources)
    for match in selected:
        gt = ground_truth(match)
        paths = _needs("config.match_paths (Member 4, M0)", config.match_paths, match, sources)
        info = data_io.video_info(paths.clip)
        frames = frameset_range(args.frameset, info['frame_count_header'], cfg)
        written = E.write_oracle_detections(gt, args.split, match, args.frameset, frames,
                                            (info["width"], info["height"]), info["fps"], cfg,
                                            OUTPUTS, args.frozen_test)
        for detector, path in written.items():
            print(f"  {detector}: {path.relative_to(ROOT).as_posix()}")


def cmd_patches(args, cfg) -> None:
    path = OUTPUTS / "scores" / args.detector / args.split / "patches.csv"
    matches(args, cfg)  # test-split guard
    doc = E.evaluate_patches(path, cfg)
    save([doc])
    log_test(args, args.detector, [doc])


def _per_match(args, cfg, kind: str, evaluate) -> None:
    docs, runs = [], []
    for match in matches(args, cfg):
        path = artefact(kind, args.detector or args.pipeline, args.split, match, args.frameset)
        gt = ground_truth(match)
        docs.append(evaluate(path, gt, cfg))
        runs.append((path, gt))
    pooled = None
    if len(docs) > 1:
        ap = E.pooled_detection_ap(runs, cfg) if kind == "detections" else None
        pooled = E.pool_metrics(docs, ap=ap)
    save(docs, pooled)
    log_test(args, args.detector or args.pipeline, docs + ([pooled] if pooled else []))


def cmd_proposals(args, cfg) -> None:
    _per_match(args, cfg, "proposals", E.evaluate_proposals)


def cmd_detection(args, cfg) -> None:
    _per_match(args, cfg, "detections", E.evaluate_detections)


def cmd_tracking(args, cfg) -> None:
    contracts.split_pipeline_id(args.pipeline)
    _per_match(args, cfg, "tracks", E.evaluate_tracking)


def cmd_compare(args, cfg) -> None:
    from soccertrack import compare

    written = compare.write_comparison(OUTPUTS / "metrics", args.split, OUTPUTS / "tables",
                                       OUTPUTS / "figures" / "eval", OUTPUTS / "eda" / "stats.json")
    for path in written:
        print(f"  {path.relative_to(ROOT).as_posix()}")


def cmd_fixtures(args, cfg) -> None:
    """Run E1-E4, the oracle writer and the comparison on tests/fixtures (no data needed).

    Writes to outputs/fixtures_check/ so fixture numbers never mix with real results in outputs/metrics/.
    """
    from soccertrack import compare

    fx = ROOT / "tests" / "fixtures"
    out = OUTPUTS / "fixtures_check"
    gt = contracts.read_table(fx / "tracks_gt.csv", "tracks")
    truth = "tests/fixtures/tracks_gt.csv"
    docs = [E.evaluate_patches(fx / "patch_scores_sample.csv", cfg),
            E.evaluate_proposals(fx / "proposals_sample.csv", gt, cfg, ground_truth=truth),
            E.evaluate_detections(fx / "detections_sample.csv", gt, cfg, ground_truth=truth),
            E.evaluate_tracking(fx / "tracks_pred.csv", gt, cfg, workspace_root=out / "trackeval", ground_truth=truth)]
    oracle = E.write_oracle_detections(gt, "val", "999001", "fixture3", range(0, 3), (640, 360), 25.0, cfg, out)
    docs += [E.evaluate_detections(path, gt, cfg, ground_truth=truth) for path in oracle.values()]
    for doc in docs:
        E.write_metrics(doc, out / "metrics")
        shown = ", ".join(f"{k}={v}" for k, v in doc["metrics"].items() if v is not None)
        print(f"  {doc['run_id']}\n    {shown}")
    written = compare.write_comparison(out / "metrics", "val", out / "tables", out / "figures", fx / "eda_stats_sample.json")
    print(f"\nTables and figures in {(out).relative_to(ROOT).as_posix()}/ ({len(written)} files)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from soccertrack import split
    split.require_test_access(args.split, args.frozen_test)
    if args.command in ("patches", "proposals", "detection") and not args.detector:
        build_parser().error(f"{args.command} needs --detector")
    if args.command == "tracking" and not args.pipeline:
        build_parser().error("tracking needs --pipeline <detector>+<tracker>")
    if args.command == "failures":
        print("'failures' is Phase 3 (failure analysis on train/val); not part of Phase 2.")
        return 2
    cfg = E.load_evaluation_config()
    if args.command != 'fixtures':
        matches(args, cfg)  # includes compare; no dispatch/path reads before authorization
    if args.detector is not None and args.detector not in contracts.DETECTOR_IDS:
        raise contracts.ContractError('Unknown detector ID')
    if args.pipeline is not None:
        contracts.split_pipeline_id(args.pipeline)
    print(f"[{args.command}] split={args.split} frameset={args.frameset}")
    try:
        globals()[f"cmd_{args.command}"](args, cfg)
    except SetupMissing as exc:
        print(f"BLOCKED: {exc}")
        return 3
    except (contracts.ContractError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
