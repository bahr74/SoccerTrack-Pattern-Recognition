# SoccerTrack — Vision-Based Football Player Detection, Tracking and External Workload Analysis

**Course:** SBES 426, Medical Pattern Recognition · **Dataset:** SoccerTrack v2 (release v1.2) · **Deadlines:** Phase 2 Wed 25 Nov 2026 · Phase 3 Sat 19 Dec 2026

> **Status: integrated Phase 2 validation implemented and executed.** The accepted local package is `outputs/phase2_final/phase2_20261125_01/START_HERE.md`. Launch the single application with `.venv/Scripts/python.exe demos/phase2/launch.py`. The unified source notebook is `notebooks/phase2_baseline.ipynb`; the report source is `docs/phase2/report/phase2_report.md`. Existing checkpoints and YOLO cache are reused. Custom HOG reference verification remains environment-blocked; browser visual acceptance is manual. Phase 3 and the final presentation remain pending. Generated scientific outputs are intentionally ignored by Git; obtain the accepted package or reproduce explicitly with the existing approved data/models. See the package for exact tests, limitations and artifact hashes.

---

## 1. What this project does
- **Goal:** detect and track the 22 players in fixed panoramic football video (one 4-minute clip per match, 25 fps, ~4096 × 1080 px, median player box ≈ 20 × 45 px).
- **Phase 3:** derive **external workload indicators** (trajectories, distance, speed, acceleration, heatmaps). These are external-load proxies; the dataset has no physiological fatigue or injury ground truth, and none is claimed.
- **Two complete, independent pipelines** are compared under one shared evaluation:

```
                      ┌─────────── Pipeline A: from scratch (Member 1) ───────────┐
 clip ─► data layer ─►│ segmentation → region proposals → HOG+SVM | Small CNN → NMS │─► Detection table ─► IoU + Hungarian tracker ─► Track table ─┐
 (0-based frames)     └────────────────────────────────────────────────────────────┘                                                              │
                      ┌──────────── Pipeline B: pretrained (Member 2) ─────────────┐                                                              ├─► evaluation E1–E4 (Member 3)
                 ────►│ native-resolution tiles → COCO YOLOv8n → merge + NMS        │─► Detection table ─► ByteTrack ─────────────► Track table ─┘     ↓
                      └────────────────────────────────────────────────────────────┘                                          tables, figures, notebook, report (Members 3, 4)
                                                                                          Phase 3: Track table → pitch mapping → workload (Member 4)
```

- The two trackers are separate implementations.
- All modules communicate only through the file contracts in [`docs/team/SHARED_CONTRACTS.md`](docs/team/SHARED_CONTRACTS.md) (Detection, Track, Proposal and Patch tables; run metadata; Metrics JSON).

## 2. Team roles

| Member | Responsibility | Task guide | AI prompt | Branch |
|---|---|---|---|---|
| **1** | Data loading, cleaning, EDA, split; segmentation, proposals, HOG+SVM, Small CNN, scratch detector; IoU + Hungarian tracker | [MEMBER_1_SCRATCH.md](docs/team/MEMBER_1_SCRATCH.md) | [prompt](docs/team/prompts/MEMBER_1_AI_PROMPT.md) | `feature/scratch-pipeline` |
| **2** | Tiled YOLOv8n detector, box merging, ByteTrack; Phase 3 fine-tuning | [MEMBER_2_YOLO.md](docs/team/MEMBER_2_YOLO.md) | [prompt](docs/team/prompts/MEMBER_2_AI_PROMPT.md) | `feature/yolo-pipeline` |
| **3** | Contracts; classification, detection and tracking evaluation; comparison; Phase 3 failure analysis and explainability | [MEMBER_3_EVALUATION.md](docs/team/MEMBER_3_EVALUATION.md) | [prompt](docs/team/prompts/MEMBER_3_AI_PROMPT.md) | `feature/evaluation` |
| **4** | Config and data checks; visualisation; notebooks, report, slides; integration; Phase 3 pitch mapping and workload | [MEMBER_4_VISUALIZATION.md](docs/team/MEMBER_4_VISUALIZATION.md) | [prompt](docs/team/prompts/MEMBER_4_AI_PROMPT.md) | `feature/visualization-docs` |

**File ownership:** every file has exactly one owner ([`INTEGRATION_PLAN.md`](docs/team/INTEGRATION_PLAN.md) §2). Edit only your own files; request changes to others through their owner.

## 3. Quick start for a new collaborator (Windows, PowerShell in VS Code)

1. **Clone** into a folder on a drive with a few GB free (`D:\` is only an example):
   ```powershell
   cd D:\
   git clone https://github.com/bahr74/SoccerTrack-Pattern-Recognition.git pattern_prj
   cd pattern_prj
   ```
2. **Find your role** in the table above and open your task guide at **§0 START HERE**, a step-by-step first session. Read, in order:
   1. [`SETUP_AND_DATA.md`](docs/team/SETUP_AND_DATA.md) §1–3: what the dataset is and which data you need (usually none at first);
   2. your task guide;
   3. [`SHARED_CONTRACTS.md`](docs/team/SHARED_CONTRACTS.md);
   4. [`INTEGRATION_PLAN.md`](docs/team/INTEGRATION_PLAN.md) (§0 is the team execution sequence);
   5. the technical plan [`phase2_phase3_plan.md`](docs/planning/phase2_phase3_plan.md) (Revision 3.1).
3. **Environment.** Only after the team has approved installation. Details and troubleshooting are in [`SETUP_AND_DATA.md`](docs/team/SETUP_AND_DATA.md).
   ```powershell
   $Drive = "D:"   # a drive that exists on YOUR computer and has ~8 GB free (SETUP_AND_DATA.md §4.1)
   $env:PIP_CACHE_DIR = "$Drive\pip_cache"; $env:TMP = "$Drive\tmp"; $env:TEMP = "$Drive\tmp"; $env:HF_HOME = "$Drive\hf_cache"
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install -e . --no-deps
   ```
4. **Run the fixture tests.** No dataset, no GPU and no `configs/data_sources.yaml` needed:
   ```powershell
   pytest            # contract + unit tests on tests/fixtures/
   pytest -rs        # also lists the skipped acceptance tests that are still to be implemented
   ```
5. **Create your branch** (after the skeleton is on `main`; see [`GIT_WORKFLOW.md`](docs/team/GIT_WORKFLOW.md)):
   ```powershell
   git switch main; git pull origin main
   git switch -c feature/<your-branch>
   ```
6. **Implement one milestone at a time.** Open your AI prompt (`docs/team/prompts/MEMBER_<n>_AI_PROMPT.md`), paste it into your coding assistant, and work only on your files. Replace the milestone's skipped tests with real ones.
7. **Review and submit:**
   ```powershell
   git status
   git diff --cached --name-only
   pytest
   git commit
   git push
   ```
   Then open a pull request into `main` with the checklist from `GIT_WORKFLOW.md` §5.

## 4. Data and models (never committed)
- **Source:** SoccerTrack v2, Hugging Face `atomscott/soccertrack-v2` (gated), revision `325bd4c57ee8e135ab652f3921d274526ada496b`.
  - Licence: CC BY 4.0. Cite the dataset and the paper (arXiv 2508.01802).
- **Split:**
  - train: 118575, 118576, 118577, 118578, 128058 (117092 optional);
  - val: 117093;
  - test: 128057, 132831.
  - The test set is held out and evaluated once per phase with a frozen configuration.
- **Locations:**
  - On the lead machine, train/val files and the COCO YOLOv8n weights are referenced **read-only** from an existing local folder.
  - The two test matches (~0.91 GB) are downloaded only after approval.
  - Every machine has its own git-ignored `configs/data_sources.yaml`, created from [`configs/data_sources.example.yaml`](configs/data_sources.example.yaml).
  - Members without the local data work from `tests/fixtures/` (Mode 1) or download only the files their role needs (Mode 2). See [`SETUP_AND_DATA.md`](docs/team/SETUP_AND_DATA.md) §2–3 (data modes and per-member data plan), §5.4 (approved YOLO weights) and §6 (local paths).
- **Official results** are produced on the data machine from `main` ([`INTEGRATION_PLAN.md`](docs/team/INTEGRATION_PLAN.md) §4).

## 5. Repository layout

| Path | Content |
|---|---|
| `src/soccertrack/` | The Python package (one module per responsibility; owners in each module docstring) |
| `scripts/` | One runner per member (`run_scratch.py`, `run_yolo.py`, `run_evaluation.py`, `run_visualization.py`) + `check_data.py`, `download_test_matches.py` |
| `configs/` | `project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml` (one owner each) + the data-path template |
| `tests/` | `conftest.py`, synthetic `fixtures/` with hand-computed expected values, one test file per module |
| `notebooks/` | `phase2_baseline.ipynb`, `phase3_final.ipynb` (only Member 4 edits them) |
| `docs/planning/` | Technical plan (Revision 3.1) and data transfer manifest |
| `docs/team/` | Contracts, integration plan, Git workflow, setup, member guides, AI prompts |
| `docs/phase2/`, `docs/phase3/` | Report sections per member, curated figures, final report and slides |
| `docs/phase1/` | Phase 1 deliverables (frozen) |
| `data/`, `outputs/` | Local, git-ignored data and generated artefacts |

## 6. Phase boundaries

| Phase 2 (due 25 Nov) | Phase 3 (due 19 Dec) |
|---|---|
| EDA, preprocessing, leakage-safe split; segmentation + proposals; HOG+SVM and Small CNN; scratch detector; tiled YOLOv8n (zero-shot); E1–E3; one frozen test run; *should-have:* both trackers with E4 on oracle boxes and detections | Failure analysis on train/val; improvements (YOLO fine-tuning, scratch v2, tracker v2); explainability; full-clip tracking; final frozen test run; pitch mapping and external workload indicators |

## 7. Workflow and rules
- **Roles:** planning was done with Claude Code, reviewed by the team and ChatGPT. Members implement their milestones with their AI prompt.
- **Integration:** all integration happens through pull requests into `main`; nobody pushes to `main` directly.
- **Leakage and data:** no test-set tuning, no unapproved downloads, no copies of datasets or weights, no credentials in any file.

## 8. References
- SoccerTrack v2 code: https://github.com/AtomScott/SoccerTrack-v2
- Dataset: https://huggingface.co/datasets/atomscott/soccertrack-v2
- Paper: https://arxiv.org/abs/2508.01802
