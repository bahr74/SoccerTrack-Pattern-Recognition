# Phase 2 Report Section — Member 4: Configuration, Visualisation & Integration

Owner: Member 4. Only Member 4 edits this file.

Topics: Dataset description, EDA figures, overlays, integration notes, limitations.

---

## 1. Configuration Foundation (M0)

### Method
`config.py` provides a centralised, machine-independent interface to the project's four committed YAML config files (`project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`). Machine-specific absolute paths are read exclusively from the git-ignored `configs/data_sources.yaml`. Placeholder detection (`<ST_LOCAL>`, `<REPO>`) ensures members do not accidentally commit partial configs.

Key functions:
- `load_config()` — loads all four committed config sections via `yaml.safe_load`.
- `load_data_sources()` — validates placeholders and raises a helpful error if missing.
- `artifact_path(kind, method, split, match_id, frameset)` — builds canonical output paths following SHARED_CONTRACTS §2 (e.g. `outputs/detections/yolov8n_coco/val/117093__eval10.csv`) and enforces `read_only_roots`.
- `frameset_range(name, num_frames, cfg)` — resolves frameset definitions from `evaluation.yaml`; `stop=null` → clip length.
- `config_sha256()` — SHA-256 over the four committed configs with normalised line endings; used as the frozen-test provenance hash.
- `set_seeds(seed)` — seeds Python `random`, NumPy, and PyTorch with deterministic cuDNN.

### Settings
Seed: `42` (from `configs/project.yaml`). Config SHA-256 is recomputed at every frozen test run and stored in `outputs/metrics/*.json → provenance.config_sha256`.

### Results
All 5 acceptance tests in `tests/test_config.py` pass on fixtures (no real data needed). The data-marked test (`test_data_01`) is deferred to the data machine.

### Limitations
- `artifact_path` does not validate the `kind` or `method` strings against the closed vocabulary; that check belongs to `contracts.validate_*`.
- `config_sha256` does not cover `data_sources.yaml` by design (machine-independence).

---

## 2. Data Verification Script (M0)

### Method
`scripts/check_data.py` reads `data_sources.yaml` and verifies:
- File existence and byte size.
- MOT annotation: 10 columns, 0-based frames (min frame must be 0).
- Video header via `cv2.VideoCapture`: width, height, fps, container frame count.
- Optional SHA-256 against a manifest file.
- Optional full sequential decode (via `data_io.count_decoded_frames`) with `--decode`.

Output: `outputs/data_check.json` and a console summary table.

### Limitations
Full SHA-256 verification and frame-count checks require real data files and are therefore deferred to the data machine.

---

## 3. EDA Figures (M1)

### Method
`visualize.plot_eda(stats, box_sample, out_dir)` produces five figures from the EDA stats JSON (SHARED_CONTRACTS §11):

1. **Boxes per match** — bar chart coloured by split (train/val/test).
2. **Box size distribution** — p05/median/p95 for width and height per match, with the paper's reference median overlaid.
3. **Height vs. image row** — scatter of box height against foot y-coordinate, illustrating the perspective effect.
4. **Track length distribution** — min/median/max frames per match (all 22 tracks are continuous in SoccerTrack v2).
5. **Data quality issues** — table of degenerate boxes, clipped boxes, duplicate rows, and frames with ≠ 22 boxes.

All figures are saved at 150 DPI (from `configs/project.yaml → visualization.figure_dpi`).

### Settings
Figures rendered from `tests/fixtures/eda_stats_sample.json` during development; replaced by `outputs/eda/stats.json` once Member 1 delivers M2.

### Results
> **[PLACEHOLDER]** Real EDA stats not yet available. Fixture-based figures confirm the rendering pipeline is correct. Replace with real per-match numbers from `outputs/eda/stats.json`.

Key findings from the fixture (and expected to hold on real data):
- Exact 22 boxes per frame in clean clips.
- 0-based frame indices confirmed.
- All four trailing MOT columns are `−1`.

### Limitations
- `height_vs_row.png` requires `box_sample.csv`; skipped if the file is absent.
- Paper comparison is approximate (our clips are slightly smaller cameras).

---

## 4. Box Overlays and Video (M1/M2)

### Method
`visualize.draw_boxes(frame, table, frame_id)`:
- Draws boxes from a Track or Detection table onto a BGR frame.
- **Stable per-track colour**: `hash(track_id * 137 % 360)` → HSV → BGR; the same player always gets the same colour across frames.
- Never modifies the input (always works on a copy).
- Labels show `track_id` for Track tables, `score` for Detection tables.

`visualize.side_by_side(frame, left, right, frame_id, titles)`:
- Stacks two overlays **vertically** (not side-by-side), because panoramic frames are 4096 × 1080 — horizontal stacking would produce a 4096 × 2160 image that is impractical to display.
- Titles are drawn in a filled rectangle for readability.

`visualize.overlay_video(clip_path, tables, frames, out_path, scale=0.5)`:
- Writes an MP4 at 0.5× scale (4096 × 1080 → 2048 × 540) for the presentation.
- Uses `data_io.iter_frames` for sequential decoding; falls back to a direct `cv2` loop if not yet available.

### Results
> **[PLACEHOLDER]** Real overlays require decoded frames from Member 1 (M1 `iter_frames`). Fixture-based tests pass. Real overlays will be produced after Member 1's M1 is merged.

### Limitations
- Real overlays depend on Member 1's `data_io.iter_frames`.
- Overlay video is git-ignored (`*.mp4`); a curated frame PNG is committed to `docs/phase2/figures/`.

---

## 5. Stage Figure (M2)

### Method
`visualize.plot_stage_figure(stage_dir, frame_id, out_path)` composes four images from Member 1's `outputs/stages/<match>/` into a 2×2 panel: original frame / foreground mask / post-morphology / region proposals.

### Results
> **[PLACEHOLDER]** Waiting for Member 1's segmentation outputs (`outputs/stages/`).

---

## 6. Integration Notes

- `config.py` is the **first module to be merged** (M0); all other members import it.
- `artifact_path` is the single source of truth for output file locations; all runners call it.
- The notebook (`phase2_baseline.ipynb`) operates in **fast mode** by default: it reads saved `outputs/` artefacts and displays placeholders where artefacts are missing, without crashing.
- Each section of the notebook is populated from the corresponding `docs/phase2/sections/member_*.md` file.

---

## 7. Limitations (Phase 2)

- Real EDA figures, overlays, and the stage figure all depend on Member 1's data processing milestones.
- The comparison overlay depends on both Members 1 and 2 producing their track tables.
- The frozen test run (and all test-split metrics) requires the two test matches to be downloaded and the pipeline to be fully integrated on the lead machine.
- Notebook full-mode execution requires all artefacts to exist in `outputs/`.

