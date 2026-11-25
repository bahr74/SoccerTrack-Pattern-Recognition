# Member 4 — Phase 2 Work Summary & Technical Explanation

**Role:** Configuration, Visualisation, Documentation & Integration
**Branch:** `feature/visualization-docs`
**Phase:** Phase 2 (Baseline Player Detection and Tracking on SoccerTrack v2)

This document summarises all code, scripts, tests, notebook integration, and documentation implemented by Member 4 from the start of Phase 2, including post-merge integration with Member 2 (`feature/yolo-pipeline`) and Member 3 (`feature/evaluation`).

---

## 1. Overview of Completed Work

| Milestone | Scope | Files Modified | Status |
|---|---|---|---|
| **M0** | Configuration loading, path resolution, data verification, test-match download script | `src/soccertrack/config.py`, `scripts/check_data.py`, `scripts/download_test_matches.py`, `tests/test_config.py` | Complete & tested |
| **M1** | Bounding-box overlay drawing and EDA plotting | `src/soccertrack/visualize.py`, `tests/test_visualize.py` | Complete & tested on fixtures |
| **M2** | Side-by-side comparison overlay, overlay MP4 writer, segmentation stage grid, CLI runner | `src/soccertrack/visualize.py`, `scripts/run_visualization.py`, `tests/test_visualize.py` | Complete & tested on fixtures |
| **M3** | Phase 2 baseline notebook (fast mode) & contract integration | `notebooks/phase2_baseline.ipynb`, `scripts/run_visualization.py` | Complete & executed cleanly |
| **M4 (Section)** | Member 4 Phase 2 report section draft | `docs/phase2/sections/member_4_visualization.md` | Draft complete (placeholders await real data runs) |

---

## 2. Detailed Explanation by Component

### 2.1 Configuration & Provenance Foundation (`src/soccertrack/config.py`)
Every module in `src/soccertrack/` imports `config.py` as the single source of configuration, paths, and reproducibility provenance:

- **`repo_root() -> Path`**: Resolves the repository root relative to `config.py`'s own file location (`Path(__file__).resolve().parent.parent.parent`), so callers work regardless of their current working directory (important for `nbconvert` and scripts).
- **`load_config() -> dict[str, Any]`**: Loads the four committed YAML configuration files (`project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`) via `yaml.safe_load` and raises a clear `FileNotFoundError` if any is missing.
- **`load_data_sources(path=None) -> dict[str, Any]`**: Loads the git-ignored `configs/data_sources.yaml` (or a custom path for testing). Raises `FileNotFoundError` pointing to `configs/data_sources.example.yaml` if missing, and raises `ValueError` if any `<ST_LOCAL>` or `<REPO>` placeholder remains unfilled.
- **`match_paths(match_id, sources) -> MatchPaths`**: Returns a frozen `MatchPaths` dataclass (`match_id`, `mot`, `clip`, `calibration_dir`) from `data_sources.yaml`.
- **`weights_path(key, sources) -> Path`**: Resolves approved weight file paths by key (e.g. `yolov8n_coco`).
- **`frameset_range(name, num_frames, cfg) -> range`**: Reads named framesets (`eval10`, `win60s`, `full`, `smoke`, `fixture3`) from `configs/evaluation.yaml`, clamping `stop=null` or out-of-bounds `stop` values to `num_frames`.
- **`artifact_path(kind, method, split, match_id, frameset, *, sources=None) -> Path`**: Builds canonical contract paths (`outputs/<kind>/<method>/<split>/<match_id>__<frameset>.csv`), creates parent directories automatically, and raises `PermissionError` if the resolved path falls inside any `read_only_roots` entry.
- **`config_sha256() -> str`**: Computes a deterministic SHA-256 hex digest over the four committed YAML files in `CONFIG_FILES` order, normalising `\r\n` and `\r` to `\n` so Windows and Linux produce identical hashes. `data_sources.yaml` is excluded so frozen-test hashes are machine-independent.
- **`set_seeds(seed)`**: Seeds Python `random`, NumPy, and PyTorch (if installed), and enables deterministic cuDNN mode when CUDA is available.

### 2.2 Data Verification & Download Scripts
- **`scripts/check_data.py`**:
  - Verifies MOT annotation files (`_check_mot`: existence, byte size, 10 comma-separated columns, 0-based frame indices, annotated frame count).
  - Verifies video headers (`_check_video`: width, height, fps, container frame count via `cv2.VideoCapture`).
  - Supports `--decode` (uses Member 1's `data_io.count_decoded_frames` when available, with a sequential `cv2` fallback) to compare decoded frame counts against annotated frame counts.
  - Supports optional SHA-256 verification against a configured manifest (`--no-hash` to skip).
  - Writes a structured JSON report to `outputs/data_check.json` and prints a console summary table.
- **`scripts/download_test_matches.py`**:
  - Supports `--dry-run` to list the 13 held-out test match files (`128057`, `132831`, plus optional `132877` with `--with-val-optional`) and approximate sizes (~911.1 MB) without touching the network.
  - For real downloads, enforces that `HF_HOME` is set and not on `C:`, and downloads each explicit file via `huggingface_hub.hf_hub_download()` at the pinned revision `325bd4c57ee8e135ab652f3921d274526ada496b` (never snapshotting the full 145 GB dataset).

### 2.3 Visualisation Module (`src/soccertrack/visualize.py`)
Consumes contract tables and EDA/stage outputs without modifying evaluation metrics or input arrays:

- **`draw_boxes(frame_bgr, table, frame_id, color_by="track_id", label=True)`**: Copies `frame_bgr`, filters rows for `frame_id`, assigns a stable HSV→BGR colour per `track_id` (or green fallback when `color_by` is absent, such as in Detection tables), and draws boxes with `track_id` or `score` labels.
- **`plot_eda(stats, box_sample, out_dir)`**: Generates up to 5 PNG figures at 150 DPI:
  1. `boxes_per_match.png` — annotated box counts per match coloured by split.
  2. `box_size_hist.png` — p05/median/p95 width and height per match with paper reference medians.
  3. `height_vs_row.png` — box height vs. foot y-coordinate scatter (when `box_sample` is provided).
  4. `track_length.png` — min/median/max track length per match.
  5. `data_issues.png` — per-match quality issues table (highlighting non-zero counts).
- **`side_by_side(frame_bgr, left, right, frame_id, titles)`**: Renders two overlays and stacks them **vertically** (`np.vstack`) with title banners, keeping 4096×1080 panoramic frames legible.
- **`overlay_video(clip_path, tables, frames, out_path, scale=0.5)`**: Decodes frames sequentially via `data_io.iter_frames` (with a `cv2` fallback), overlays all tables, scales by `scale` (default 0.5 → 2048×540), and writes an MP4 at 25 fps.
- **`plot_stage_figure(stage_dir, frame_id, out_path)`**: Composes the four segmentation stage images (`frame`, `foreground`, `morph`, `proposals`) into a 2×2 PNG figure, inserting clean placeholders for missing panels.
- **Phase 3 pitch helpers**: Implemented `plot_trajectories()`, `plot_heatmap()`, and `_draw_pitch()` for 105 × 68 m pitch-space visualisation.

### 2.4 Visualisation CLI (`scripts/run_visualization.py`)
Provides subcommands `eda`, `overlays`, `compare-overlay`, `stages`, and Phase 3 stubs `pitch` / `workload`:
- `eda --use-fixtures` renders EDA figures directly from `tests/fixtures/eda_stats_sample.json`.
- `overlays` and `compare-overlay` validate input tables using Member 3's `contracts.read_table(path, kind)` before rendering.

### 2.5 Phase 2 Baseline Notebook (`notebooks/phase2_baseline.ipynb`)
Populated all 14 sections in **fast mode** (no algorithm logic inside notebook cells; reads saved `outputs/` artefacts and falls back cleanly to `tests/fixtures/` with `[PLACEHOLDER]` notices when upstream runs are not yet present):
- Uses `config.repo_root()` so execution succeeds regardless of working directory.
- Integrates Member 3's exact `compare.py` output paths (`outputs/tables/detection_val.csv`, `outputs/tables/tracking_val.csv`, `outputs/tables/references.csv`, `outputs/figures/eval/pr_detection_val.png`, `outputs/figures/eval/recall_by_height_val.png`) and Metrics JSON glob patterns.

### 2.6 Unit & Contract Tests
Replaced all placeholder `@pytest.mark.skip` tests in `tests/test_config.py` and `tests/test_visualize.py` and added coverage for the CLI helpers:
- **`tests/test_config.py`**: `test_01` (4 config sections & disjoint splits), `test_02` / `test_02b` (placeholder & missing-file errors), `test_03` / `test_03_readonly_guard` (`artifact_path` format & read-only root protection), `test_04` (`frameset_range` clamping), `test_05` (`config_sha256` determinism & line-ending stability), `test_06_check_data_helpers` (MOT 10-col & 0-based checks), `test_07_download_dry_run` (`--dry-run` output without network). (`test_data_01` remains `@pytest.mark.data` for the data machine.)
- **`tests/test_visualize.py`**: `test_01` (`draw_boxes` shape, immutability, visible drawing), `test_02` (`plot_eda` on `eda_stats_sample.json`), `test_03` (`side_by_side` vertical stacking), `test_04_plot_stage_figure` (2×2 stage figure with placeholders).

---

## 3. Post-Merge Integration & Verification

After merging `main` (which included Member 3's `feature/evaluation` PR #3 and Member 2's `feature/yolo-pipeline` PR #4) into `feature/visualization-docs` with zero conflicts:

- **`pytest tests/test_config.py tests/test_visualize.py tests/test_contracts.py -v`**: **52 passed, 1 deselected** (`@pytest.mark.data`), **0 skipped**.
- **`jupyter nbconvert --to notebook --execute notebooks/phase2_baseline.ipynb --output-dir outputs/notebook_check`**: **exited 0** (committed notebook remains cleared of outputs).
- **`python scripts/run_visualization.py eda --use-fixtures`**: **exited 0**, generating 4 PNGs in `outputs/figures/eda/`.
- **`python scripts/download_test_matches.py --dry-run`**: **exited 0**, listing 13 files (~911.1 MB).
