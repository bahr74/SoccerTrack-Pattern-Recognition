# AI implementation prompt — Member 4 (Configuration, Visualisation, Documentation, Integration; Phase 3 Workload)

> Paste everything below the line into Claude Code, Codex or another coding assistant, opened in the repository root `D:\pattern_prj` (or your clone) on the branch `feature/visualization-docs`.

---

You are implementing **Member 4's** part of a university Pattern Recognition project (SBES 426, Medical Pattern Recognition): *Vision-Based Football Player Detection, Tracking and External Workload Analysis* on **SoccerTrack v2** (Hugging Face `atomscott/soccertrack-v2`, release v1.2, revision `325bd4c57ee8e135ab652f3921d274526ada496b`).

## Project in brief
- **Two complete, independent pipelines** detect and track the 22 players in panoramic clips (~4096 × 1080 px, 25 fps):
  - **Pipeline A (Member 1):** from-scratch detector + IoU + Hungarian tracker.
  - **Pipeline B (Member 2):** tiled YOLOv8n + ByteTrack.
  - **Member 3** owns contracts and evaluation.
- **Your role:**
  - **Foundation:** configuration loading and paths, data verification, the test-match download script.
  - **Visualisation:** overlays, EDA and segmentation figures.
  - **Deliverables:** documented notebooks, report and slides, integration.
  - **Phase 3:** pitch mapping and external workload indicators (distance, speed, acceleration, heatmaps). These are external-load proxies, never physiological fatigue or injury measures.

## FIRST SESSION (do this before writing any code)

The repository documents and `docs/team/SHARED_CONTRACTS.md` are authoritative. If this prompt and a document disagree, follow the document and tell me.

1. **Check where you are.** Run `git branch --show-current` (must be `feature/visualization-docs`) and `git status`. If you are on `main`, stop and tell me.
2. **Read the dataset orientation** `docs/team/SETUP_AND_DATA.md` §1 (what SoccerTrack v2 files are, 0-based frames, xywh boxes, BGR vs RGB, split and leakage rules) and §2–3 (data modes; my row). Then read `docs/team/MEMBER_4_VISUALIZATION.md` §0 (START HERE, dependency matrix) and the files listed below.
3. **Inspect the repository.** Open my owned skeleton files and their tests, then run `pytest -rs tests/test_config.py` to see the current skips. Do not change anything yet.
4. **Summarise in at most 10 lines** before coding: the milestone, the files you will touch, the skips you will replace, the shared functions you depend on and how you will work around them. Wait for my "go".
5. **First milestone:** **M0 only:** `config.py`, `scripts/check_data.py` (tested on fake files in `tmp_path`), `scripts/download_test_matches.py --dry-run` (no network), and a README onboarding check. Replace the M0 skips in `tests/test_config.py`.
6. **Shared functions that may still be placeholders** (they raise `NotImplementedError`): `data_io.count_decoded_frames` / `iter_frames` / `load_gt_tracks` (Member 1), `contracts.*` (Member 3), all comparison outputs (Member 3). Check with `Select-String -Path src\soccertrack\<file>.py -Pattern NotImplementedError`. Never implement, copy or edit another member's function.
7. **While a dependency is missing:** use `tests/fixtures/`, the `synthetic_frame` / `synthetic_patches` fixtures, `tmp_path` and `monkeypatch` mocks. Mark tests that need real data `@pytest.mark.data` and GPU tests `@pytest.mark.gpu`. Tell me which dependency blocks which part.
8. **Data access for this milestone:** Mode 1 only. Use the committed configs and temporary files; the dry run must not use the network. Never download data or weights, never log in to Hugging Face, never install packages: ask me.
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
2. `docs/team/SHARED_CONTRACTS.md` (Sections 1, 2, 10–13 matter most for you)
3. `docs/team/MEMBER_4_VISUALIZATION.md` (your task guide)
4. `docs/team/INTEGRATION_PLAN.md` (you own it), `docs/team/SETUP_AND_DATA.md`, `docs/team/GIT_WORKFLOW.md`
5. Skeletons: `src/soccertrack/{__init__,config,visualize,pitch,workload}.py`, `scripts/{run_visualization,check_data,download_test_matches}.py`, `configs/project.yaml`, `configs/data_sources.example.yaml`, `tests/test_{config,environment,visualize,pitch_workload}.py`, `notebooks/*.ipynb`

## Files you may create or edit (nothing else)
- `src/soccertrack/`: `__init__.py`, `config.py`, `visualize.py`, `pitch.py`, `workload.py`
- `scripts/run_visualization.py`, `scripts/check_data.py`, `scripts/download_test_matches.py`
- `configs/project.yaml` (**never change the split values** without approval from all members), `configs/data_sources.example.yaml`
- `tests/test_config.py`, `tests/test_environment.py`, `tests/test_visualize.py`, `tests/test_pitch_workload.py`
- `notebooks/phase2_baseline.ipynb`, `notebooks/phase3_final.ipynb`
- `docs/phase2/**`, `docs/phase3/**` except other members' `sections/` files
- `docs/team/*` except `SHARED_CONTRACTS.md` (team-agreed changes only)
- `README.md`, `requirements.txt` (pin versions after an approved install), `pyproject.toml`, `.gitignore`

## Files you must NOT modify
- **Other members' code:** `contracts.py`, `metrics.py`, `evaluate.py`, `compare.py`, `explain.py`, `data_io.py`, `split.py`, `segmentation.py`, `proposals.py`, `patches.py`, `features.py`, `classifiers.py`, `detector_scratch.py`, `tracker_iou.py`, `detector_yolo.py`, `tracker_bytetrack.py`.
- **Other configs and tests:** `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`, `tests/conftest.py`, `tests/fixtures/**`, other tests.
- **Other docs:** other members' `sections/` files, `SHARED_CONTRACTS.md`, `docs/planning/**`, `docs/phase1/**`.
- **Data:** anything under `F:\`.

## Interfaces you use (read/import only)
- `soccertrack.contracts`: `read_table`, `read_run_meta`, `validate_*`, `write_table` (Phase 3 PitchTrack/Workload), constants.
- `soccertrack.data_io`: `iter_frames`, `video_info`, `load_gt_tracks`, `count_decoded_frames`.
- **Member 3's outputs:** `outputs/tables/*.csv`, `outputs/figures/eval/*.png`, `outputs/metrics/*.json`. Never recompute metrics.

## What to implement
1. **`config.py` (milestone M0: highest priority, everyone imports it):**
   - `repo_root` (from the file location);
   - `load_config` → `{"project","scratch","yolo","evaluation"}`;
   - `load_data_sources`: validate structure; reject unfilled `<ST_LOCAL>`/`<REPO>` placeholders with a message pointing to the template;
   - `match_paths` → `MatchPaths`; `weights_path`;
   - `frameset_range` (stop null → num_frames; clamp);
   - `artifact_path(kind, method, split, match_id, frameset)` → `outputs/<kind>/<method>/<split>/<match>__<frameset>.csv` (create parents; refuse any path below a `read_only_roots` entry);
   - `config_sha256` (four committed YAML files in fixed order, line endings normalised; excludes `data_sources.yaml`);
   - `set_seeds`.
2. **`scripts/check_data.py`:**
   - for each match of the chosen split(s): files exist; byte size and (unless `--no-hash`) SHA-256 against the optional manifest;
   - MOT has 10 columns and min frame 0 (minimal read of the txt; no dependency on `data_io` for this);
   - video header width/height/fps via `cv2`; with `--decode`, `data_io.count_decoded_frames` must equal max(frame) + 1;
   - weights exist.
   - Writes `outputs/data_check.json` and prints a table. Read-only for every input.
3. **`scripts/download_test_matches.py`:**
   - `huggingface_hub.hf_hub_download(repo_id="atomscott/soccertrack-v2", repo_type="dataset", revision=<pinned>, filename=<each explicit file>, local_dir=<repo>/data/raw/soccertrack-v2)`;
   - only `mot/README.md`, `mot/{128057,132831}.txt`, `mot/clips/{128057,132831}.mp4` and the small `raw/{128057,132831}/` calibration files; `--with-val-optional` adds 132877;
   - `--dry-run` lists the pinned files, sizes and destinations from the list in the script, without network access and without downloading;
   - refuse to run if `HF_HOME` is unset or its drive lacks free space for the download (on the lead machine C: is nearly full); never read, print or store tokens (the user logs in with `hf auth login` or `huggingface-cli login`, depending on the installed version).
   - **Do not run the real download yourself.** Only `--dry-run` is allowed until I confirm team approval.
4. **`visualize.py`:**
   - `draw_boxes` (stable colour per track_id; does not modify input);
   - `side_by_side` (A vs B stacked vertically);
   - `overlay_video` (MP4 under `outputs/overlays/`, scaled 0.5, frames from `data_io.iter_frames`);
   - `plot_eda` (boxes per match, width/height histograms, height vs image row, track lengths, issues table) from `outputs/eda/stats.json` and `box_sample.csv`;
   - `plot_stage_figure` (from Member 1's stage PNGs);
   - Phase 3: `plot_trajectories`, `plot_heatmap`.
   - Figures with matplotlib at `visualization.figure_dpi`; outputs under `outputs/figures/{eda,overlays,workload}/`.
5. **`scripts/run_visualization.py` commands:** `eda` (with `--use-fixtures`), `overlays`, `compare-overlay`, `stages`, (Phase 3) `pitch`, `workload`.
6. **Notebooks:**
   - Fill `phase2_baseline.ipynb` following its section headers.
   - **Fast mode** reads `outputs/` (metrics JSON, tables, figures); **full mode** calls the runner scripts. No algorithm code in notebooks.
   - Text comes from `docs/phase2/sections/member_*.md`. Clear outputs before committing.
7. **Report and slides:**
   - Order: preprocessing and split → baseline architectures and training settings → evaluation results and comparison with the reference paper → limitations and Phase 3 plan.
   - Every number is traceable to a Metrics JSON `run_id` or a table in `outputs/tables/`.
8. **Phase 3:**
   - `pitch.py`: load keypoints + intrinsics; undistort; RANSAC homography; leave-one-out error in metres; foot point = bottom-centre; `quality` flag; verify the axis convention.
   - `workload.py`: Savitzky–Golay smoothing; speed/acceleration at 25 fps; distance; high-speed (≥ 5.5 m/s) and sprint (≥ 7.0 m/s) distance; acceleration events (|a| ≥ 3 m/s²); summary table per the contract.
   - Validation: GT tracks vs predicted tracks through the same mapping.

## Milestones: implement in this order and STOP after each one to report
- **M0:** `config.py` + `check_data.py` + download script (dry run) + README setup section (merge first).
- **M1:** `draw_boxes` + EDA figures (fixtures first, then real `stats.json`).
- **M2:** overlays, A-vs-B side-by-side, stage figure, short MP4.
- **M3:** notebook in fast mode; smoke test after each integration step.
- **M4:** report and slides.
- **M5:** Phase 2 checklist and submission.
- **Phase 3:** pitch mapping → workload → final deliverables.

## Data and model paths (lead machine)
- **Train/val (read-only):**
  - `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\<id>.txt` and `...\mot\clips\<id>.mp4`;
  - validation clip `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4`;
  - optional 117092 in `F:\Football-Fatigue-Project\data\raw\soccertrack_v2\mot\`.
- **Calibration:** `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\<id>\` (117092, 117093, 118575, 118576).
- **Weights:** `F:\Football-Fatigue-Project\models\yolov8n.pt`, `yolov8s.pt`.
- **Test (lead machine, after the approved download):** `<repo>\data\raw\soccertrack-v2\mot\...`.
- These absolute paths belong **only** in the git-ignored `configs/data_sources.yaml`.

## Hard rules
- **No test leakage:** visualising or analysing test results is allowed only after the logged frozen test run. Never change anything in response to test results.
- **No downloads** (except an approved, user-run test download), **no installs** without my approval, no copies of data/weights, never write to `F:\`.
- **No git commits, pushes or branch changes** unless I ask.
- **Never commit:** `configs/data_sources.yaml`, notebook outputs, MP4s, `data/`, `outputs/`. Never put tokens or credentials in any file.
- **Code quality:** documented, typed, small functions; parameters from `configs/project.yaml`; no metric computation in `visualize.py`.
- **Tests:** replace every `skip` in your test files; use `tests/fixtures/` and `conftest.synthetic_frame`; mark real-data tests `@pytest.mark.data`; run `pytest` before reporting.
- **Phase 3 compatibility:** keep the skeleton signatures; Phase 3 modules consume only contract tables.
- **Medical/sports-science honesty:** describe workload outputs as external-load indicators with calibration uncertainty (~2–3 m). Never claim fatigue or injury detection.
- **Ambiguity:** if any contract, path or ownership rule is unclear, stop and ask me.

## Validation commands (PowerShell, `.venv` active)
```powershell
pytest
pytest tests/test_config.py tests/test_visualize.py tests/test_environment.py
pytest -m data tests/test_config.py
python scripts/check_data.py --split all
python scripts/download_test_matches.py --dry-run
python scripts/run_visualization.py eda --use-fixtures
jupyter nbconvert --to notebook --execute notebooks/phase2_baseline.ipynb --output-dir outputs/notebook_check
```

## How to report after each milestone (handoff report)

Reply with exactly these sections:

```
### Handoff report — Member 4, milestone M<k>
Status: complete | partial | blocked
Changed files (owned only): <list>
Implemented: <3–6 bullets>
Tests
  - Implemented and run: <command> → <real summary: N passed, N failed>
  - Skips replaced in this milestone: <test names>
  - Skips still remaining (pytest -rs): <count, with milestone IDs>
  - Integration / data / gpu tests: <run or not run, and why>
Results: <figures produced (paths) and alignment checks>
Dependencies used / still placeholders: <function → owner → workaround used>
Deviations from the guide: <none | what and why>
Blockers / requests for other members: <none | who, what>
Next step: <the next milestone, not started>
```

Never claim tests passed without running them. Do not commit or push unless I ask; when I do, stage only owned files and show `git diff --cached --name-only` first.
