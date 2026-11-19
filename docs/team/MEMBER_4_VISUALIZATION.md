# Member 4 — Configuration, Visualisation, Documentation and Integration (Phase 3: Pitch Mapping and Workload)

Branch: `feature/visualization-docs` · AI prompt: [`prompts/MEMBER_4_AI_PROMPT.md`](prompts/MEMBER_4_AI_PROMPT.md)
Read first: [`../planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1), [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md), [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) (you own it), [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

---

## 0. START HERE — Your first working session

Your first task: **M0: `config.py` (config loading, data-path handling, framesets, artifact paths, config hash, seeds), `scripts/check_data.py`, `scripts/download_test_matches.py --dry-run`, and the README onboarding check.**

Data mode for this task: **Mode 1 only.** `test_config.py` uses the committed configs and temporary fake files; `check_data.py` is tested on files you create in `tmp_path`; the download script is tested in `--dry-run` mode, which must not touch the network (modes: [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §2; your data plan: §3).

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
   4. [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md): §1, §2 (file naming), §3–4 (tables you draw), §8 (Metrics JSON you report), §10–13;
   5. [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0, §2–4;
   6. [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §1–5.
3. **Create your branch:**
   ```powershell
   git switch main; git pull origin main
   git switch -c feature/visualization-docs
   ```
4. **Know your files:**
   - **Owned** (edit only these): `src/soccertrack/{__init__,config,visualize,pitch,workload}.py`, `scripts/{run_visualization,check_data,download_test_matches}.py`, `configs/project.yaml`, `configs/data_sources.example.yaml`, `tests/test_{config,environment,visualize,pitch_workload}.py`, notebooks, `docs/team/*` except `SHARED_CONTRACTS.md`, `docs/phase2/**`/`docs/phase3/**` except others' `sections/`, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`.
   - **Read-only** (import or read, never edit): `contracts.py`, `compare.py` outputs, `data_io.py`, other members' code, configs, tests and sections.
   - **Off-limits:** `docs/phase1/**` (frozen); `F:\Football-Fatigue-Project`; recomputing metrics (tables come from Member 3).
5. **Python environment** (only after the team has approved installation): [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §4.
6. **Initial fixture checks** (✅ work immediately, no data needed):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pytest                                  # a few contract/fixture checks pass; most tests are SKIPPED placeholders
   pytest -rs tests/test_config.py                 # shows the skipped acceptance tests of your first milestone
   python scripts/run_visualization.py --help             # argument parsing only; the commands themselves are not implemented yet
   ```
   A green run with skips proves nothing about your code. Any call into an unimplemented function raises `NotImplementedError("TODO(Member n, Mk)")`.
7. **Understand your first milestone:** §9 (milestones) and §0.2–0.3 below.
8. **Implement only that milestone.** Do not start the next one in the same branch state.
9. **Replace the skipped tests of the milestone** with real tests (remove the `@pytest.mark.skip`, write the assertions):
   - `test_config.py`: four sections loaded and split disjoint; placeholders rejected; `artifact_path` naming and read-only roots; `frameset_range`; stable `config_sha256`
   - the data-marked path check runs later on a machine with data (`pytest -m data tests/test_config.py`)
10. **Run and inspect:**
    ```powershell
    pytest tests/test_config.py -v
    pytest -rs                              # no skip with your milestone ID may remain
    ```
11. **Document:** docstrings in your modules, and a short note of what was done in `docs/phase2/sections/ (you assemble the report from all four)`.
12. **Report blockers** to the owner of the missing piece, through the team chat or the PR, never by editing their file ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §9).
13. **Commit and open a PR** (only when your team has authorised it):
    ```powershell
    git status
    git add <your owned files only>
    git diff --cached --name-only           # nothing outside your owned files; no data, weights, outputs or credentials
    git commit -m "M<k>: <what was done>"
    git push -u origin feature/visualization-docs
    ```
    Then open a pull request into `main` with the checklist in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §5.

**Commands that work only later:**

| Command | Works |
|---|---|
| `python scripts/check_data.py --split all` on real files | ⏳ after M0, on a machine with data; frame counts additionally need Member 1's `count_decoded_frames` (M1) |
| `python scripts/download_test_matches.py` (real download) | ⏳ only on the lead machine, after team approval, with the operator's own Hugging Face login |
| `python scripts/run_visualization.py eda|overlays|compare-overlay|stages …` | ⏳ after your M1/M2 (real overlays also need Member 1's `iter_frames`) |
| `jupyter nbconvert --execute notebooks/phase2_baseline.ipynb` | ⏳ after M3, with `nbconvert` installed |

### 0.1 Your data
Data needed now → source → local configuration → verification → data needed later: see your row in [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §3. Never put datasets, weights, outputs or credentials into Git.

### 0.2 Dependency and independence matrix

| Milestone | Required input | Exists now? | Owner / provider | If not ready yet, use | Proof of completion | Hand-off |
|---|---|---|---|---|---|---|
| **M0** config + data checks | Committed configs, template, temporary files | Yes | Member 4 | `data_io.count_decoded_frames` (Member 1 M1) may still raise: check existence and byte sizes now; add frame counts after M1 is merged | All non-data `test_config.py` skips replaced and passing; `check_data` tests on `tmp_path`; dry run lists the files and sizes of SETUP §5.2 | `config.py` → everyone |
| **M1** `draw_boxes`, EDA figures | `eda_stats_sample.json`, `synthetic_frame`, `tracks_gt.csv` | Yes (real `stats.json` after Member 1 M2) | Member 3 (fixtures); Member 1 | Fixtures | `test_visualize.py` skips replaced and passing | Figure functions; EDA figures → report |
| **M2** overlays, A-vs-B, stage figure, MP4 | Real frames + Detection/Track tables | After Member 1 M1 (frames) and Members 1/2 detection milestones | Members 1, 2 | Fixture tables drawn on `synthetic_frame` | Alignment check per producer (Integration Plan §5.2) | Overlay figures → report |
| **M3–M5** notebook, report, slides | Metrics JSON, comparison tables, member sections | After Member 3 M1/M4 and the real runs | Members 1–3; lead machine | `metrics_sample.json`, fixture tables; placeholders clearly marked | Notebook executes in fast mode; every number traceable | Phase 2 submission |
| **Phase 3** pitch + workload | Calibration files; full-clip Track tables | Calibration on the lead machine; tracks after Phase 3 tracking | Dataset; Members 1, 2 | `pitch_tracks_sample.csv` (4.8 m over 25 frames) | `test_pitch_workload.py` | Workload tables and figures → Phase 3 report |

Parallel work is allowed, but integration checkpoints cannot be skipped. Each milestone that others depend on is merged into `main` before they rely on it ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0 and §4).

### 0.3 Building figures and the report before models exist
- **What works from fixtures alone:**
  - EDA figures from `tests/fixtures/eda_stats_sample.json`;
  - box overlays from `tracks_gt.csv` / `detections_sample.csv` drawn on the `synthetic_frame` fixture;
  - notebook sections and report/slide templates with clearly marked placeholders.
- **What needs real inputs later:**

| Later task | Needs | From |
|---|---|---|
| Real EDA figures | `outputs/eda/stats.json` | Member 1 M2 |
| Real overlays, stage figure, short MP4 | decoded val frames + Detection/Track tables | Member 1 M1 (frames), Members 1/2 detection and tracking milestones |
| Result tables and evaluation figures in the report | Metrics JSON, comparison tables | Member 3 M1/M4, from runs on the lead machine |
| Notebook full mode | all artefacts above | lead machine |
| Phase 3 pitch maps and workload | calibration files, full-clip tracks | lead machine; Phase 3 tracking |

- You never run models and never need weights.

---

## 1. Project overview and scientific objective
- **Task:** detect and track the 22 players in SoccerTrack v2 panoramic clips. Two complete pipelines are compared under one evaluation:
  - **Pipeline A (Member 1):** from scratch, ending in an IoU + Hungarian tracker.
  - **Pipeline B (Member 2):** tiled YOLOv8n + ByteTrack.
  - **Member 3** evaluates both.
- **Your role:**
  - Keep the project **runnable and coherent**: configuration loading, data checks, test-match download script, environment.
  - Make the results **visible and understandable**: overlays, EDA and stage figures.
  - Turn everyone's work into the **deliverables**: documented notebook, report, presentation, integration checklist.
- **Phase 3:** turn tracks into **external workload indicators** (pitch mapping, distance, speed, acceleration, heatmaps), with honest uncertainty and no physiological claims.

## 2. Your scope

**Phase 2:**
1. **Foundation (M0, first):**
   - `config.py`: config loading, `data_sources.yaml` validation, artefact paths, framesets, config hash, seeds.
   - `scripts/check_data.py`.
   - `scripts/download_test_matches.py`, run by a member with their own Hugging Face token, after approval.
   - Environment set-up support.
2. Detection/track overlays (single and A-vs-B side by side), short overlay video for the presentation.
3. EDA figures from Member 1's `stats.json`; segmentation stage figure from Member 1's stage images.
4. Documented notebook `phase2_baseline.ipynb` (fast mode reads `outputs/`); report; slides.
5. Integration: smoke tests after each merge step, Phase 2 checklist, submission.

**Phase 3:**
- `pitch.py`: calibration → homography, leave-one-out error.
- `workload.py`: smoothing, speed, acceleration, speed bands, per-player summaries.
- Trajectories and heatmaps.
- GT-track vs predicted-track workload comparison.
- Final notebook, report and slides.

## 3. Algorithms, methods and concepts
- **Configuration:** YAML loading (`yaml.safe_load`); placeholder detection; absolute paths only from `data_sources.yaml`; read-only roots enforced in `artifact_path`; SHA-256 over the four committed config files with normalised line endings.
- **Verification:**
  - `hashlib.sha256` streaming;
  - `cv2.VideoCapture` header (width, height, fps, frame count);
  - optional full decode count via `data_io.count_decoded_frames` (Member 1);
  - MOT checks (10 columns, min frame 0, max frame + 1 vs decoded frames).
- **Download:**
  - `huggingface_hub.hf_hub_download(repo_id="atomscott/soccertrack-v2", repo_type="dataset", revision=<pinned>, filename=..., local_dir=...)` per explicit file;
  - `HF_HOME` on D:; never snapshot the 145 GB repository; never handle the token in code.
- **Overlays:**
  - OpenCV drawing on BGR frames from `data_io.iter_frames`;
  - stable colour per `track_id` (hash → HSV);
  - GT in one colour and predictions in another for alignment checks;
  - 4096 × 1080 panoramas are stacked vertically for A vs B and scaled 0.5 for video.
- **Figures:** matplotlib, consistent style, readable at slide size, saved at `visualization.figure_dpi`.
- **Phase 3 pitch mapping:**
  - undistort keypoints with the camera intrinsics (`cv2.undistortPoints`);
  - homography with `cv2.findHomography(RANSAC)`;
  - foot point = bottom-centre;
  - leave-one-out landmark error in metres (earlier work: median ≈ 2.6 m on 117093);
  - corrected keypoints for 132831 from the official repository's `data_corrections/` (download only after approval);
  - verify the axis convention before use.
- **Workload:**
  - Savitzky–Golay smoothing (`scipy.signal.savgol_filter`);
  - speed and acceleration by finite differences at 25 fps;
  - distance = Σ step lengths;
  - high-speed (≥ 5.5 m/s) and sprint (≥ 7.0 m/s) distance; acceleration events |a| ≥ 3 m/s²;
  - heatmaps on a 105 × 68 m grid.

## 4. Files you own (edit only these)
- `src/soccertrack/__init__.py`, `config.py`, `visualize.py`, `pitch.py`, `workload.py`
- `scripts/run_visualization.py`, `scripts/check_data.py`, `scripts/download_test_matches.py`
- `configs/project.yaml` (split values frozen), `configs/data_sources.example.yaml`
- `tests/test_config.py`, `test_environment.py`, `test_visualize.py`, `test_pitch_workload.py`
- `notebooks/phase2_baseline.ipynb`, `notebooks/phase3_final.ipynb` (**only you edit notebooks**)
- `docs/phase2/**` and `docs/phase3/**` except other members' `sections/` files
- `docs/team/*` except `SHARED_CONTRACTS.md`
- `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`

## 5. Shared files you may read/import but must not edit
- `contracts.py`, `compare.py` outputs (Member 3)
- `data_io.py` (`iter_frames`, `load_gt_tracks`, `count_decoded_frames`) (Member 1)
- other members' configs, code, tests and sections
- `docs/phase1/**` (frozen)
- You never recompute metrics: tables and evaluation figures come from Member 3.

## 6. Inputs and outputs

| You read | You write |
|---|---|
| `outputs/eda/stats.json`, `box_sample.csv` (Member 1) | `outputs/figures/eda/*.png` |
| `outputs/stages/...` (Member 1) | `outputs/figures/overlays/*.png`, `outputs/overlays/*.mp4` (git-ignored) |
| Detection/Track tables of both pipelines | `outputs/data_check.json` |
| `outputs/tables/*.csv`, `outputs/figures/eval/*.png`, Metrics JSON (Member 3) | Curated `docs/phase2/figures/*.png` (committed), notebook, report, slides |
| Phase 3: calibration files (`calibration_dir`), Track tables | `outputs/pitch/...` (PitchTrack), `outputs/workload/...` (summary), `outputs/figures/workload/*.png` |

## 7. Data access
- You create the **template** `configs/data_sources.example.yaml` (done in the skeleton) and help each member fill in their git-ignored `configs/data_sources.yaml`.
- Lead machine paths: train/val on `F:\Football-Fatigue-Project\...` (read-only); test under `<repo>\data\raw\soccertrack-v2\` (lead machine: `D:\pattern_prj`). Exact paths are in `SETUP_AND_DATA.md`.
- Calibration (Phase 3): `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\<id>\` for 117092, 117093, 118575, 118576. Test matches are downloaded with the test files.
- Without real data, develop with `tests/fixtures/` and `conftest.synthetic_frame`.

## 8. Dependencies and setup
- `opencv-python`, `matplotlib`, `pandas`, `pyyaml`, `huggingface_hub` (download script), `ipykernel`/`nbconvert` (notebook execution), `scipy` (Phase 3).
- You maintain `requirements.txt`: after the first approved install, pin the versions of `huggingface_hub`, `ipykernel` and `nbconvert`.

## 9. Milestones

| ID | Target | Deliverable | Acceptance |
|---|---|---|---|
| **M0** | Fri 20, 12:00 | `config.py`, `check_data.py`, `download_test_matches.py` (dry run), README setup section | `test_config.py` passes and `check_data.py` is unit-tested on temporary fake files (no dataset needed); foundation PR merged. The real check of the 7 local matches is run afterwards by whoever operates the data machine |
| **M1** | Fri 20 → Sat 21 | `draw_boxes`, EDA figures (fixtures first, then real `stats.json`) | `test_visualize.py`; figures render from `eda_stats_sample.json` |
| **M2** | Sat 21 → Sun 22 | Overlays: single, A-vs-B side by side, stage figure, short MP4 | GT/prediction alignment check for each new producer (Integration Plan §5.2) |
| **M3** | Sun 22 → Mon 23 | Notebook skeleton populated in fast mode from available artefacts; smoke tests after each merge | `jupyter nbconvert --execute` passes in fast mode |
| **M4** | Tue 24 | Report and slides from member sections, Metrics JSON and tables | Every number traceable; assignment order respected |
| **M5** | Wed 25 | Phase 2 checklist complete; submission | `INTEGRATION_PLAN.md` §7 all ticked or explained |

## 10. Commands (PowerShell, repository root, `.venv` active)
```powershell
pytest tests/test_config.py tests/test_environment.py tests/test_visualize.py
pytest -m data tests/test_config.py
python scripts/check_data.py --split all
python scripts/check_data.py --split val --decode
python scripts/download_test_matches.py --dry-run          # real download only after approval
python scripts/run_visualization.py eda --use-fixtures
python scripts/run_visualization.py eda
python scripts/run_visualization.py overlays --pipeline yolov8n_coco+bytetrack --split val --match 117093 --frameset win60s
python scripts/run_visualization.py compare-overlay --split val --match 117093 --frameset win60s
jupyter nbconvert --to notebook --execute notebooks/phase2_baseline.ipynb --output-dir outputs/notebook_check
```

## 11. Tests and acceptance
- Implement every `skip` in your test files.
- `test_environment.py` must pass after installation (`pytest -m gpu` for CUDA).
- Visual checks: GT and prediction boxes align on the same frame index for every producer (catches off-by-one and coordinate errors). Record the result in the PR description.

## 12. Working independently with fixtures
- EDA figures: `tests/fixtures/eda_stats_sample.json`.
- Overlays: `conftest.synthetic_frame(boxes)` + `tracks_gt.csv` / `tracks_pred.csv` / `detections_sample.csv`.
- Notebook layout: use fixture outputs until real artefacts exist.
- Phase 3 workload: `tests/fixtures/pitch_tracks_sample.csv` (constant 5 m/s → 4.8 m over 24 steps).

## 13. Expected results and deliverables
- Working foundation (config, data check, download script).
- EDA figures; stage figure; overlays (PNG + short MP4); A-vs-B side-by-side.
- Notebook (fast mode), report and slides. Phase 2 report order: preprocessing and split → baseline architectures and training settings → evaluation results and comparison with the reference paper → limitations and Phase 3 plan.
- Phase 3: per-player workload tables (GT vs predicted), trajectories, heatmaps, calibration error.

## 14. Integration requirements
- `config.py` (M0) is merged first together with Member 3's contracts and Member 1's data foundation.
- Run the smoke test (Integration Plan §5.1) on the data machine after each integration step.
- Notebook cells call `src/` functions or read `outputs/`. No algorithm code in notebooks.
- Copy curated figures into `docs/phase2/figures/` (small PNGs only).

## 15. Common mistakes and prohibited changes
- Hard-coding paths or committing `configs/data_sources.yaml`; writing anything under `F:\`.
- Downloading anything other than the approved test files; snapshotting the whole dataset; putting tokens in files or notebooks.
- Typing numbers into the report by hand instead of reading Metrics JSON/tables; recomputing metrics.
- Committing notebook outputs, MP4s or large images.
- Editing other members' code, configs or sections.
- Phase 3: calling distance/speed "fatigue" or "injury risk"; ignoring calibration error.

## 16. Definition of done
- Foundation merged and used by everyone.
- All your tests pass.
- Overlays and figures produced from real artefacts.
- Notebook executes in fast mode.
- Report and slides submitted on time.
- Phase 2 checklist complete.
- Phase 3: workload pipeline validated against GT tracks with an uncertainty statement.

## 17. Phase 3 extensions
- `pitch.py`: per-match calibration, leave-one-out error table, PitchTrack outputs for GT tracks and for the best predicted pipeline.
- `workload.py` and visualisations:
  - smoothing chosen on GT tracks of train/val;
  - per-player summaries;
  - GT vs predicted distance error (%);
  - trajectories and heatmaps.
- Optional bonus (only after the required items): 2D tactical-board animation, per-player workload cards, simple static dashboard. A Wadi Degla demo only with documented consent and team approval.
- Final notebook, report and slides; limitations (4-minute clips, ~2–3 m calibration error, no physiological ground truth).
