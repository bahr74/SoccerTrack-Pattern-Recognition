# AI implementation prompt — Member 2 (Pretrained YOLO Pipeline B)

> Paste everything below the line into Claude Code, Codex or another coding assistant, opened in the repository root `D:\pattern_prj` (or your clone) on the branch `feature/yolo-pipeline`.

---

You are implementing **Member 2's** part of a university Pattern Recognition project (SBES 426, Medical Pattern Recognition): *Vision-Based Football Player Detection, Tracking and External Workload Analysis* on the **SoccerTrack v2** dataset (Hugging Face `atomscott/soccertrack-v2`, release v1.2, revision `325bd4c57ee8e135ab652f3921d274526ada496b`).

## Project in brief
- **Data:** fixed panoramic football clips (4 minutes per match, 25 fps, ~4096 × 1080 px). 22 GT player boxes per frame; median ≈ 20 × 45 px, so whole-frame downscaling destroys the players.
- **Two complete, independent pipelines** are compared with identical evaluation:
  - **Pipeline A (another member):** from-scratch detector + IoU + Hungarian tracker.
  - **Pipeline B (you):** native-resolution tiling → COCO-pretrained YOLOv8n ("person") → global merging + NMS → ByteTrack.
- **Integration:** all modules communicate only through file contracts (CSV + JSON metadata). Another member evaluates your files.

## FIRST SESSION (do this before writing any code)

The repository documents and `docs/team/SHARED_CONTRACTS.md` are authoritative. If this prompt and a document disagree, follow the document and tell me.

1. **Check where you are.** Run `git branch --show-current` (must be `feature/yolo-pipeline`) and `git status`. If you are on `main`, stop and tell me.
2. **Read the dataset orientation** `docs/team/SETUP_AND_DATA.md` §1 (what SoccerTrack v2 files are, 0-based frames, xywh boxes, BGR vs RGB, split and leakage rules) and §2–3 (data modes; my row). Then read `docs/team/MEMBER_2_YOLO.md` §0 (START HERE, dependency matrix) and the files listed below.
3. **Inspect the repository.** Open my owned skeleton files and their tests, then run `pytest -rs tests/test_detector_yolo.py` to see the current skips. Do not change anything yet.
4. **Summarise in at most 10 lines** before coding: the milestone, the files you will touch, the skips you will replace, the shared functions you depend on and how you will work around them. Wait for my "go".
5. **First milestone:** **M1 only:** `make_tiles`, `tile_to_frame`, `merge_tile_detections` in `detector_yolo.py`, tested on synthetic sizes and boxes. Replace the M1 skips in `tests/test_detector_yolo.py`. No video, weights or GPU.
6. **Shared functions that may still be placeholders** (they raise `NotImplementedError`): `data_io.iter_frames` / `video_info` (Member 1), `split.require_test_access` (Member 1), `config.weights_path` / `artifact_path` (Member 4), `contracts.validate_detections` / `write_table` (Member 3), all of `evaluate.py` (Member 3). Check with `Select-String -Path src\soccertrack\<file>.py -Pattern NotImplementedError`. Never implement, copy or edit another member's function.
7. **While a dependency is missing:** use `tests/fixtures/`, the `synthetic_frame` / `synthetic_patches` fixtures, `tmp_path` and `monkeypatch` mocks. Mark tests that need real data `@pytest.mark.data` and GPU tests `@pytest.mark.gpu`. Tell me which dependency blocks which part.
8. **Data access for this milestone:** Mode 1 only. Do not read videos, load weights or use the GPU in this milestone. Never download data or weights, never log in to Hugging Face, never install packages: ask me.
9. **Work only in my owned files** (list below). If something outside them must change, stop and describe the change for its owner.
10. **Stop and report after each milestone** using the handoff report at the end. Never implement several milestones or the whole of Phase 2 in one run.
11. **Report tests in three separate groups:**
    - implemented tests that ran (passed/failed);
    - placeholder tests still skipped (`pytest -rs`);
    - integration, `data` or `gpu` tests: run or not run, and why.
    A green run with skips is **not** evidence that anything is implemented.

## Read these files first and follow them exactly
0. `docs/team/SETUP_AND_DATA.md` §1–3 (dataset orientation and data modes): **before implementing any module**
1. `docs/planning/phase2_phase3_plan.md` (Revision 3.1)
2. `docs/team/SHARED_CONTRACTS.md`
3. `docs/team/MEMBER_2_YOLO.md` (your task guide)
4. `docs/team/INTEGRATION_PLAN.md`, `docs/team/SETUP_AND_DATA.md`, `docs/team/GIT_WORKFLOW.md`
5. The skeletons: `src/soccertrack/detector_yolo.py`, `src/soccertrack/tracker_bytetrack.py`, `scripts/run_yolo.py`, `configs/yolo.yaml`, `tests/test_detector_yolo.py`, `tests/test_tracker_bytetrack.py`

## Files you may create or edit (nothing else)
- `src/soccertrack/detector_yolo.py`, `src/soccertrack/tracker_bytetrack.py`
- `scripts/run_yolo.py`
- `configs/yolo.yaml`
- `tests/test_detector_yolo.py`, `tests/test_tracker_bytetrack.py`
- `docs/phase2/sections/member_2_yolo.md`, `docs/phase3/sections/member_2_yolo.md`

Private helpers go inside these files. Ask me before creating any new module.

## Files you must NOT modify
- **All other code:** `src/soccertrack/*` except your two modules.
- **Other configs and tests:** other `configs/*.yaml`, `tests/conftest.py`, `tests/fixtures/**`, other tests.
- **Shared files:** notebooks, `docs/team/**`, `docs/planning/**`, `docs/phase1/**`, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`.
- **Data:** anything under `F:\`.

If a change seems necessary there, **stop and tell me**.

## Interfaces you use (read/import only)
- `soccertrack.config`: `load_config`, `load_data_sources`, `match_paths`, `weights_path`, `frameset_range`, `artifact_path`, `config_sha256`, `set_seeds`.
- `soccertrack.contracts`: `write_table`, `read_table`, `write_run_meta`, `read_run_meta`, `validate_*`.
- `soccertrack.data_io`: `iter_frames`, `video_info`, `load_gt_tracks`.
- `soccertrack.split`: `matches_for`, `require_test_access`, `log_test_access`.

If one is not implemented yet:
- use fixtures or test-local helpers **inside your test files**;
- keep production code calling the official interface;
- report the missing dependency.

**Never** import `tracker_iou`, `detector_scratch`, `classifiers` or other Pipeline A code.

## Conventions that are not negotiable
- **Frames:** 0-based, sequentially decoded (`data_io.iter_frames`). Never seek by time.
- **Detection table:** `frame,x,y,w,h,score`, where `x, y` is the **top-left** corner and `w, h` the size in **full-frame** pixels.
  - Ultralytics returns tile-local xyxy: add the tile offset, merge across tiles, clip to the image, convert to xywh.
  - Keep **all** detections with `score ≥ score_floor` (0.05, `configs/evaluation.yaml`). The operating threshold is chosen on validation and stored in the metadata.
- **Track table:** `frame,track_id,x,y,w,h,score`; metadata `tracker: "bytetrack"`.
- **Paths:**
  - detections: `outputs/detections/<detector>/<split>/<match>__<frameset>.csv` + `.meta.json`
  - tracks: `outputs/tracks/<detector>+bytetrack/<split>/<match>__<frameset>.csv` + `.meta.json`
  - Use `config.artifact_path`. Debug runs with `--max-frames` go to `outputs/debug/` only.
- **IDs:** detector `yolov8n_coco` (Phase 3: `yolov8n_ft`, `yolov8s_ft`); tracker `bytetrack` (Phase 3: `bytetrack_tuned`).
- **Framesets:** from `configs/evaluation.yaml` (`eval10`, `win60s`, `full`, `smoke`).

## Algorithms to implement (details in MEMBER_2_YOLO.md §3)
1. **`make_tiles(W, H, tile_w, tile_h, overlap)`:** full coverage, overlap ≥ `overlap_px`, last tile flush with the edge; works for 4096 × 1080, 3840 × 1504 and frames smaller than a tile.
2. **`tile_to_frame`, `merge_tile_detections`:** offsets; `torchvision.ops.nms(global_nms_iou)` for cross-tile duplicates.
3. **`YoloTiledDetector`:**
   - Load weights with `ultralytics.YOLO(<absolute path from config.weights_path("yolov8n_coco")>)`. First check that the file exists and raise a clear error if not. **Never** pass a bare `"yolov8n.pt"` or a missing path: Ultralytics then downloads weights.
   - If my machine has no F: drive, do not obtain weights yourself. Use the mocked model in tests and tell me; I follow `docs/team/SETUP_AND_DATA.md` §5.4 (approved copy outside the repository, SHA-256 verified).
   - Predict all tiles of a frame in one batch: `imgsz`, `conf=conf_floor`, `iou=iou_nms_in_tile`, `classes=[0]`, `device`, `verbose=False`, `max_det`.
   - Input is BGR `uint8` NumPy.
4. **`apply_pitch_filter`:** use `outputs/masks/<match>_pitch.png` when `merge.pitch_filter` is true and the file exists. Otherwise skip and set the metadata note `pitch_filter=off`.
5. **`run_detection`:** iterate the frameset; write the Detection table + metadata including `runtime_seconds`, `frames_per_second`, `device`, `operating_threshold`.
6. **ByteTrack adapter (verified for ultralytics 8.4.182):**
   - `from ultralytics.trackers.byte_tracker import BYTETracker`; `BYTETracker(args)` with `args = types.SimpleNamespace(**cfg["yolo"]["bytetrack"] minus output_box)` having `track_high_thresh`, `track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`, `fuse_score`.
   - `update(results, img=None)` requires `results` with NumPy `conf`, `cls` and `xywh` (**centre** x, centre y, w, h), boolean indexing and `len()`. Implement `DetectionsForByteTrack` for this.
   - The output rows are `[x1, y1, x2, y2, track_id, score, cls, idx]`. With `output_box: detection`, write the input detection `idx` box; otherwise convert the Kalman xyxy box.
   - Call `update` for **every** frame of the stride-1 frameset, including empty ones.
   - Validate on `gt_oracle`, then `gt_noisy`, then YOLO detections.
7. **Validation experiments (val only):** first reproduce the earlier reference setting (1280-px tiles, `imgsz` 960, conf 0.40), then tile width 960 vs 1280, `imgsz` 960 vs 1280, overlap 64 vs 128, operating threshold, ByteTrack thresholds. Record a small experiment table in your report section.

## Milestones: implement in this order and STOP after each one to report
- **M1:** tiling + merging (CPU tests).
- **M2:** single-frame detector on one real val frame (GPU test).
- **M3:** `run_detection` on val `eval10` and `win60s` + FPS + pitch filter.
- **M4:** ByteTrack adapter (oracle → noisy → YOLO).
- **M5:** validation experiments and chosen settings in `yolo.yaml`.
- **M6:** test artefacts **only** when I explicitly say the frozen test run has started.

## Data and model paths
- From `configs/data_sources.yaml` (git-ignored). On the data machine:
  - validation clip: `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4`
  - weights: `F:\Football-Fatigue-Project\models\yolov8n.pt` (Phase 3 also `yolov8s.pt`)
  - All read-only.
- Test clips (128057, 132831): `<repo>\data\raw\soccertrack-v2\mot\clips\` on the lead machine, used only in the frozen run.
- Write only under `outputs/` and (Phase 3) `data/processed/yolo_tiles/`.

## Hard rules
- **No test leakage:** every choice (tiles, thresholds, ByteTrack parameters, fine-tuning epochs) is made on **val** 117093. Test clips only with `--frozen-test`, after my instruction, logged via `split.log_test_access`.
- **No downloads** (data or weights) and **no installs** without my approval. No copies of weights or clips in the repository. Never write to `F:\`.
- **No git commits, pushes or branch changes** unless I ask.
- **Code quality:**
  - documented, typed, small functions; parameters from `configs/yolo.yaml` only; seeds via `config.set_seeds`;
  - no new dependencies (no `supervision`, no other trackers).
- **Tests:**
  - replace every `skip` in your test files;
  - mock `ultralytics.YOLO` in CPU tests;
  - mark real-GPU tests `@pytest.mark.gpu` and real-data tests `@pytest.mark.data`;
  - run `pytest` before reporting.
- **Phase 3 compatibility:** keep the skeleton signatures; add fine-tuning in `finetune()` and new IDs; keep `yolov8n_coco` reproducible.
- **Ambiguity:** if anything in the contracts or interfaces is unclear, **stop and ask me**. Never invent conventions.

## Validation commands (PowerShell, `.venv` active)
```powershell
pytest
pytest tests/test_detector_yolo.py tests/test_tracker_bytetrack.py
pytest -m gpu tests/test_detector_yolo.py
python scripts/run_yolo.py detect --detector yolov8n_coco --split val --match 117093 --frameset smoke
python scripts/run_yolo.py track  --detector gt_oracle    --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py detection --detector yolov8n_coco --split val --frameset eval10   # Member 3's evaluator
```

## How to report after each milestone (handoff report)

Reply with exactly these sections:

```
### Handoff report — Member 2, milestone M<k>
Status: complete | partial | blocked
Changed files (owned only): <list>
Implemented: <3–6 bullets>
Tests
  - Implemented and run: <command> → <real summary: N passed, N failed>
  - Skips replaced in this milestone: <test names>
  - Skips still remaining (pytest -rs): <count, with milestone IDs>
  - Integration / data / gpu tests: <run or not run, and why>
Results: <FPS and validation numbers with `run_id`s (from M3 on)>
Dependencies used / still placeholders: <function → owner → workaround used>
Deviations from the guide: <none | what and why>
Blockers / requests for other members: <none | who, what>
Next step: <the next milestone, not started>
```

Never claim tests passed without running them. Do not commit or push unless I ask; when I do, stage only owned files and show `git diff --cached --name-only` first.
