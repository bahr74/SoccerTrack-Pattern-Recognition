# Member 2 — The Complete Pretrained YOLO Pipeline (Pipeline B)

Branch: `feature/yolo-pipeline` · AI prompt: [`prompts/MEMBER_2_AI_PROMPT.md`](prompts/MEMBER_2_AI_PROMPT.md)
Read first: [`../planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1), [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md), [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md), [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

---

## 0. START HERE — Your first working session

Your first task: **M1: tiling, tile-to-frame coordinate conversion and cross-tile NMS** (`make_tiles`, `tile_to_frame`, `merge_tile_detections`), tested on synthetic frame sizes and boxes.

Data mode for this task: **Mode 1 only.** M1 needs no video, no weights and no GPU: it is pure box arithmetic on synthetic boxes (NMS uses `torchvision.ops.nms`, which runs on CPU) (modes: [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §2; your data plan: §3).

1. **Clone and open** (only after the approved skeleton is merged into `main`):
   ```powershell
   git clone https://github.com/bahr74/SoccerTrack-Pattern-Recognition.git pattern_prj
   cd pattern_prj
   code .
   ```
2. **Read, in this order:**
   1. [`README.md`](../../README.md);
   2. [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §1–3 (dataset orientation and your data plan);
   3. this guide §0, §2–5 and §9;
   4. [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md): §1 (conventions), §4 (Detection table), §3 (Track table), §7 (run metadata), §13, §14;
   5. [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0, §2–4;
   6. [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §1–5.
3. **Create your branch:**
   ```powershell
   git switch main; git pull origin main
   git switch -c feature/yolo-pipeline
   ```
4. **Know your files:**
   - **Owned** (edit only these): `src/soccertrack/detector_yolo.py`, `src/soccertrack/tracker_bytetrack.py`, `scripts/run_yolo.py`, `configs/yolo.yaml`, `tests/test_detector_yolo.py`, `tests/test_tracker_bytetrack.py`, your two `sections/` files.
   - **Read-only** (import or read, never edit): `config.py`, `contracts.py`, `metrics.py`, `evaluate.py`, `data_io.py`, `split.py`, other configs, fixtures, notebooks, `docs/team/*`.
   - **Off-limits:** `tracker_iou.py`, `detector_scratch.py`, `classifiers.py` (never import them: Pipeline B stays independent); `docs/phase1/**`; `F:\Football-Fatigue-Project`.
5. **Python environment** (only after the team has approved installation): [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §4.
6. **Initial fixture checks** (✅ work immediately, no data needed):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pytest                                  # a few contract/fixture checks pass; most tests are SKIPPED placeholders
   pytest -rs tests/test_detector_yolo.py                 # shows the skipped acceptance tests of your first milestone
   python scripts/run_yolo.py --help             # argument parsing only; the commands themselves are not implemented yet
   ```
   A green run with skips proves nothing about your code. Any call into an unimplemented function raises `NotImplementedError("TODO(Member n, Mk)")`.
7. **Understand your first milestone:** §9 (milestones) and §0.2–0.3 below.
8. **Implement only that milestone.** Do not start the next one in the same branch state.
9. **Replace the skipped tests of the milestone** with real tests (remove the `@pytest.mark.skip`, write the assertions):
   - `test_detector_yolo.py`: 4096 × 1080 coverage with overlap and a flush last tile; 3840 × 1504 and frames smaller than one tile; `tile_to_frame` round trip; one box kept for a player duplicated across a tile border
10. **Run and inspect:**
    ```powershell
    pytest tests/test_detector_yolo.py -v
    pytest -rs                              # no skip with your milestone ID may remain
    ```
11. **Document:** docstrings in your modules, and a short note of what was done in `docs/phase2/sections/member_2_yolo.md`.
12. **Report blockers** to the owner of the missing piece, through the team chat or the PR, never by editing their file ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §9).
13. **Commit and open a PR** (only when your team has authorised it):
    ```powershell
    git status
    git add <your owned files only>
    git diff --cached --name-only           # nothing outside your owned files; no data, weights, outputs or credentials
    git commit -m "M<k>: <what was done>"
    git push -u origin feature/yolo-pipeline
    ```
    Then open a pull request into `main` with the checklist in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §5.

**Commands that work only later:**

| Command | Works |
|---|---|
| `pytest -m gpu tests/test_detector_yolo.py` (one real frame) | ⏳ after your M2, Member 1's `iter_frames` (M1), Member 4's `config` (M0) and approved local weights (SETUP §5.4) |
| `python scripts/run_yolo.py detect …` | ⏳ after M3 |
| `python scripts/run_yolo.py track …` | ⏳ after M4 |

### 0.1 Your data
Data needed now → source → local configuration → verification → data needed later: see your row in [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §3. Never put datasets, weights, outputs or credentials into Git.

### 0.2 Dependency and independence matrix

| Milestone | Required input | Exists now? | Owner / provider | If not ready yet, use | Proof of completion | Hand-off |
|---|---|---|---|---|---|---|
| **M1** tiling, mapping, NMS | Synthetic frame sizes and boxes | Yes (you create them in the test) | — | Nothing missing | M1 skips replaced and passing | Merged tiling functions (used by your M2) |
| **M2** one real frame | Decoded val frame; `yolov8n.pt`; `config.weights_path`; GPU | Frame: after Member 1 M1. Weights: lead machine F:, else SETUP §5.4. Config: after Member 4 M0 | Members 1, 4; lead-machine owner (weights) | Mock `ultralytics.YOLO` with `monkeypatch` and use the `synthetic_frame` fixture to test box conversion | `pytest -m gpu`: > 10 person boxes in full-frame coordinates; overlay checked with Member 4 | — |
| **M3** `run_detection` on val | `config.artifact_path`, `contracts.write_table`, `split.require_test_access`; Member 3's evaluator for E3 | After the foundation merge (Members 1, 3, 4); E3 after Member 3 M1 | Members 1, 3, 4 | Validate your CSV header manually against Contracts §4 until `validate_detections` exists | Output validates; E3 on val near the earlier reference | `yolov8n_coco` Detection tables + FPS → Member 3 (E3), Member 4 (overlays) |
| **M4** ByteTrack | Detection tables (`gt_oracle`, yours) | Oracle tables after Member 3 M2 | Member 3 | `tests/fixtures/detections_sample.csv` | Tracker unit tests; E4 on val via Member 3 | `<detector>+bytetrack` Track tables → Members 3, 4 |

Parallel work is allowed, but integration checkpoints cannot be skipped. Each milestone that others depend on is merged into `main` before they rely on it ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0 and §4).

### 0.3 From M1 to real inference
- **Before real inference you need:**
  - Member 1's `iter_frames` (M1) and Member 4's `config` (M0) merged into `main`, then pulled into your branch (`GIT_WORKFLOW.md` §6);
  - one validation clip (lead machine: Mode 3; otherwise Mode 2: `mot/clips/117093.mp4`, 304,980,773 bytes);
  - verified weights.
- **Weights are not shared automatically.** `yolov8n.pt` exists only on the lead machine's F: drive. Your computer does not have it just because the lead machine does. Approval-gated route ([`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §5.4):
  1. ask the team for approval;
  2. receive the 6.5 MB file through the team's private shared drive, or download it manually in the browser from the official Ultralytics assets release;
  3. store it **outside the repository**;
  4. check `Get-FileHash … -Algorithm SHA256` = `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`;
  5. set `weights: yolov8n_coco:` in your git-ignored `configs/data_sources.yaml`.
- **Never** pass a bare `"yolov8n.pt"` or a missing path to `YOLO()`: Ultralytics would download it silently. Check that the file exists first.
- **Order after M1:**
  - M2 one real frame (GPU);
  - M3 `run_detection` on val `eval10`/`win60s` + FPS;
  - M4 ByteTrack in its own module `tracker_bytetrack.py`. It is independent of Member 1's tracker and starts from `gt_oracle`, then YOLO detections;
  - M5 validation experiments;
  - M6 frozen test run.

---

## 1. Project overview and scientific objective
- **Task:** detect and track the 22 players in SoccerTrack v2 panoramic clips (release v1.2, 25 fps, ~4096 × 1080 px, median player box ≈ 20 × 45 px). The long-term goal is external workload indicators (not physiological fatigue).
- **Two complete pipelines** are compared under identical evaluation:
  - **Pipeline A (Member 1):** from scratch. Segmentation → proposals → HOG+SVM / CNN → IoU + Hungarian tracker.
  - **Pipeline B (yours):** native-resolution tiling → COCO-pretrained **YOLOv8n** ("person") → global box merging + NMS → **ByteTrack**.
- **Scientific question for you:** how well does a general-purpose pretrained detector transfer zero-shot to tiny players in a wide panorama, what does tiling cost and gain, and how does a strong standard tracker behave with these detections? Your pipeline is the strong reference that Pipeline A is compared against. In Phase 3, transfer learning (fine-tuning) is the main expected improvement.

## 2. Your scope

**Phase 2:**
1. Tiling of full-resolution frames with overlap (never downscale the whole 4096 px frame).
2. Zero-shot COCO YOLOv8n inference per tile (class 0 "person"), with a low confidence floor.
3. Tile → frame coordinate mapping, cross-tile duplicate removal (global NMS), optional pitch-mask filter.
4. Detection tables for val (`eval10`, `win60s`) with runtime/FPS in the metadata.
5. ByteTrack adapter (Ultralytics `BYTETracker`), validated first on `gt_oracle`, then `gt_noisy`, then YOLO detections → Track tables. This is a should-have for the end-to-end E4, but the adapter is in your core scope.
6. Validation experiments (tile size, overlap, floor and operating threshold, ByteTrack thresholds) and a short experiment log.
7. Test-split artefacts in the frozen run; your report section.

**Phase 3:**
- Fine-tune YOLOv8n/s on **train** tiles with early stopping on **val** (`yolov8n_ft`, `yolov8s_ft`).
- ByteTrack tuning (`bytetrack_tuned`).
- Full-clip tracking on validation, then the final frozen test.

## 3. Algorithms, methods and concepts
- **Tiling:**
  - `make_tiles(W, H, tile_w, tile_h, overlap)` places tiles at stride `tile_w − overlap`, with the last tile flush with the right/bottom edge.
  - The default is 1280-wide tiles over the full height (1080 px) with ≥ 64 px overlap.
  - It must work for other geometries, e.g. 3840 × 1504 (expected for test match 132831; verify with `check_data.py`).
- **Inference:**
  - `ultralytics.YOLO(weights).predict(tile_list, imgsz=1280, conf=conf_floor, iou=iou_nms_in_tile, classes=[0], device=0, verbose=False)`.
  - Input frames are BGR `uint8` (Ultralytics accepts BGR NumPy arrays).
  - Batch the tiles of one frame.
- **Merging:** shift boxes by the tile origin → concatenate → `torchvision.ops.nms(global_nms_iou)` → clip to the image → convert xyxy to xywh (top-left, contract convention).
- **Pitch filter (optional, `merge.pitch_filter`):** drop boxes whose bottom-centre is outside `outputs/masks/<match>_pitch.png` (Member 1's artefact). If the file is missing, run without the filter and write `notes: "pitch_filter=off"`.
- **ByteTrack** (verified for ultralytics 8.4.182, `ultralytics/trackers/byte_tracker.py`):
  - `BYTETracker(args)` with `args` having `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `fuse_score` (use `types.SimpleNamespace` filled from `configs/yolo.yaml`).
  - `update(results, img=None)` needs an object exposing NumPy `conf`, `cls`, `xywh` (**centre** x, centre y, w, h) and supporting boolean indexing and `len()`. Build this from one frame of the Detection table (`DetectionsForByteTrack`).
  - It returns rows `[x1, y1, x2, y2, track_id, score, cls, idx]`. With `output_box: detection`, write the input detection box `idx`; otherwise write the Kalman box.
  - Call `update` for **every** frame of the stride-1 frameset, including frames with no detections (pass an empty object).
  - Low-score detections (between `track_low_thresh` and `track_high_thresh`) are used in the second association stage. This is why detections down to `score_floor` must be kept.
- **Runtime:** measure wall-clock seconds and frames per second for detection (decode excluded and included; report both in `notes`).

## 4. Files you own (edit only these)
- `src/soccertrack/detector_yolo.py`
- `src/soccertrack/tracker_bytetrack.py`
- `scripts/run_yolo.py`
- `configs/yolo.yaml`
- `tests/test_detector_yolo.py`, `tests/test_tracker_bytetrack.py`
- `docs/phase2/sections/member_2_yolo.md`, `docs/phase3/sections/member_2_yolo.md`

## 5. Shared files you may read/import but must not edit
- `config.py` (Member 4)
- `contracts.py`, `metrics.py`, `evaluate.py` (Member 3)
- `data_io.py` (`iter_frames`, `video_info`, `load_gt_tracks`) and `split.py` (`require_test_access`, `log_test_access`) (Member 1)
- all configs except `yolo.yaml`; fixtures; docs; notebooks
- **Never** import `tracker_iou.py`, `detector_scratch.py` or `classifiers.py`: Pipeline B must stay independent.

## 6. Inputs and outputs

| You read | You write |
|---|---|
| Clips via `config.match_paths` + `data_io.iter_frames` | `outputs/detections/yolov8n_coco/<split>/<match>__<frameset>.csv` + `.meta.json` (§4, §7) |
| Weights `config.weights_path("yolov8n_coco")` → `F:\Football-Fatigue-Project\models\yolov8n.pt` (read-only) | `outputs/tracks/<detector>+bytetrack/<split>/<match>__<frameset>.csv` + `.meta.json` (§3) |
| `configs/yolo.yaml`, `configs/evaluation.yaml` (framesets, `score_floor`) | `outputs/debug/...` for `--max-frames` runs (never under `outputs/detections`) |
| Optional `outputs/masks/<match>_pitch.png` (Member 1) | Phase 3: `outputs/models/yolov8n_ft/`, fine-tuning tiles under `data/processed/yolo_tiles/` |
| Oracle detections `outputs/detections/gt_oracle|gt_noisy/...` (Member 3) | Experiment log in your report section |

## 7. Data access
- Configure `configs/data_sources.yaml` (see `SETUP_AND_DATA.md`).
- **Validation clip 117093:** `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4` (note the unusual folder).
- **Weights (lead machine):** `F:\Football-Fatigue-Project\models\yolov8n.pt` (and `yolov8s.pt` for Phase 3). Code must never download weights: `YOLO("yolov8n.pt")` with a bare name **downloads** a copy, so always pass the existing absolute path from `config.weights_path`.
- Test clips (128057, 132831) are under `<repo>\data\raw\soccertrack-v2\mot\clips\` on the lead machine after the approved download, used only with `--frozen-test`.
- Without F:, develop with fixtures and synthetic frames, or download only the val clip (≈ 0.30 GB) to `data/raw/` as described in `SETUP_AND_DATA.md`.
- **Weights without F:** follow `SETUP_AND_DATA.md` §5.4: mocked model for CPU tests; otherwise an approved team copy or official asset stored **outside** the repository (e.g. `<Drive>:\models\yolov8n.pt`), verified against SHA-256 `f59b3d83…fc83b36`, and referenced only in your git-ignored `configs/data_sources.yaml`.

## 8. Dependencies and setup
- `ultralytics==8.4.182`, `torch==2.9.1+cu126`, `torchvision==0.24.1+cu126`, `lap==0.5.13` (ByteTrack assignment), from `requirements.txt`. No other tracker packages (e.g. `supervision`) without team approval.
- Check the GPU with `pytest -m gpu tests/test_environment.py`.

## 9. Milestones

| ID | Target | Deliverable | Acceptance |
|---|---|---|---|
| **M1** | Fri 20 | `make_tiles`, `tile_to_frame`, `merge_tile_detections` | Unit tests on synthetic boxes (4096 × 1080 and 3840 × 1504; border duplicates) |
| **M2** | Fri 20 | `YoloTiledDetector.detect_frame` on one real frame | `pytest -m gpu`: more than 10 person boxes in full-frame coordinates; overlay check with Member 4 |
| **M3** | Sat 21 | `run_detection` for val `eval10` and `win60s`; FPS; pitch filter | Outputs validate; E3 on val (via Member 3's runner) lies near the earlier reference or the difference is investigated. The reference used 1280-px tiles at `imgsz` 960, conf 0.40, pitch-masked (P 0.846, R 0.780, 124 ms/frame); with `imgsz` 1280 it measured P 0.777, R 0.846, 162 ms/frame. Reproduce one of these settings first |
| **M4** *(should)* | Sun 22 | ByteTrack adapter | `gt_oracle+bytetrack` on `win60s` has few or no ID switches; `yolov8n_coco+bytetrack` E4 on val |
| **M5** | Sun 22 | Validation experiments: tile 960/1280, `imgsz` 960/1280, overlap 64/128, operating threshold, ByteTrack thresholds | Experiment table in your section; chosen values in `yolo.yaml` |
| **M6** | Mon 23 | Frozen test run | One logged run; section written |

## 10. Commands (PowerShell, repository root, `.venv` active)
```powershell
pytest tests/test_detector_yolo.py tests/test_tracker_bytetrack.py
pytest -m gpu tests/test_detector_yolo.py
python scripts/run_yolo.py detect --detector yolov8n_coco --split val --match 117093 --frameset smoke
python scripts/run_yolo.py detect --detector yolov8n_coco --split val --frameset eval10
python scripts/run_yolo.py detect --detector yolov8n_coco --split val --frameset win60s
python scripts/run_yolo.py track  --detector gt_oracle    --split val --frameset win60s
python scripts/run_yolo.py track  --detector yolov8n_coco --split val --frameset win60s
python scripts/run_evaluation.py detection --detector yolov8n_coco --split val --frameset eval10
python scripts/run_evaluation.py tracking  --pipeline yolov8n_coco+bytetrack --split val --frameset win60s
```

## 11. Tests and acceptance
- Implement every `skip` in your two test files; mock `ultralytics.YOLO` in non-GPU tests (return fixed boxes per tile).
- Contract tests: Detection/Track outputs pass `contracts.validate_*`; scores ≥ `score_floor`; frames inside the frameset.
- The ByteTrack oracle test on `tracks_gt.csv` (as detections) must show no ID switch.

## 12. Working independently with fixtures
- Tiling and merging need only NumPy arrays: write synthetic boxes that straddle tile borders.
- Use `conftest.synthetic_frame` for end-to-end detector tests with a mocked model.
- ByteTrack: feed `tests/fixtures/tracks_gt.csv` (drop `track_id`, score 1.0) and `detections_sample.csv`.
- The first real-data run needs only the val clip and `data_io.iter_frames` (Member 1, M1, Fri). Until then, decode with a local test helper inside your **test** file only. Do not add a second video reader to production code.

## 13. Expected results and deliverables
- **Indicative only:** earlier work on this validation match measured tiled COCO YOLOv8n at about P 0.85 / R 0.78 (F1 ≈ 0.81) and 0.12–0.16 s per frame on this GPU. Main false positives were unlabelled people (referees, staff, spectators); main misses were crowded players, players < 35 px tall and distant goalkeepers. These numbers are references, never your results.
- Deliverables: val and test Detection and Track tables with metadata; experiment table; report section.

## 14. Integration requirements
- Same contracts as Pipeline A: full-frame xywh, 0-based frames, all detections ≥ `score_floor`, `operating_threshold` chosen on val and written in the metadata.
- `tracker` metadata = `bytetrack`; pipeline folder = `<detector>+bytetrack`.
- Record the pitch-filter state in the metadata notes.

## 15. Common mistakes and prohibited changes
- Downscaling whole frames to 640/1280 (players vanish); forgetting the tile offset; duplicate boxes at tile borders.
- Writing xyxy or centre-format boxes to the Detection table (the contract is top-left xywh).
- Dropping detections below the operating threshold (breaks AP and ByteTrack's second stage).
- Skipping empty frames when calling ByteTrack (breaks track ageing).
- Passing `"yolov8n.pt"` by name or a non-existent path (both trigger a download); copying weights into the repository.
- Tuning on test; changing thresholds after the frozen run.
- Editing other members' files or configs; importing the scratch tracker.

## 16. Definition of done
- Tests implemented and passing (CPU tests always; GPU tests on the data machine).
- Val + test Detection tables and E3; ByteTrack Track tables and E4 on oracle and YOLO detections (should-have).
- FPS reported.
- Experiment table and report section complete.
- Merged without touching non-owned files.

## 17. Phase 3 extensions
- **Fine-tuning:**
  - Build 640 × 640 native-resolution tiles from **train** clips (every `label_frame_stride` frames) with YOLO-format labels from GT.
  - Keep negatives (tiles without players).
  - Train from the local `yolov8n.pt`/`yolov8s.pt`; early-stop on **val** tiles.
  - Save to `outputs/models/yolov8n_ft/`; register as `yolov8n_ft` / `yolov8s_ft`.
  - Reuse the same tiled inference code.
- **ByteTrack tuning** (`bytetrack_tuned`) on full validation clips, guided by Member 3's ID-switch analysis.
- Keep `yolov8n_coco` runnable for the Phase 2 → Phase 3 Δ table.
- If the local 4 GB GPU is insufficient, use Colab/Kaggle only after team approval, with the same code and config.
