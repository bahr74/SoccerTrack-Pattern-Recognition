# Environment and Data Setup (Windows, PowerShell)

**Owner of this document:** Member 4.
- Do not install packages or download anything until the team has approved it (`INTEGRATION_PLAN.md` §0).
- Read Section 1 **before** implementing any module. Every AI prompt requires this.

Contents:
1. Dataset orientation (read first)
2. Three data operating modes
3. Your data plan by member
4. Python environment
5. Data and weights: locations, sizes and access
6. Local path configuration (`configs/data_sources.yaml`)
7. Running tests
8. Common setup failures

---

## 1. Dataset orientation (read first)

### 1.1 What SoccerTrack v2 is
- **SoccerTrack v2:** a public football dataset recorded with a **fixed panoramic camera** that sees the whole pitch.
  - Dataset: https://huggingface.co/datasets/atomscott/soccertrack-v2 (gated: each user accepts the terms).
  - Paper: https://arxiv.org/abs/2508.01802. Official code: https://github.com/AtomScott/SoccerTrack-v2.
  - Licence: CC BY 4.0.
- **Pinned version:** release **v1.2**, revision `325bd4c57ee8e135ab652f3921d274526ada496b`. Everyone uses exactly this revision.
- **What we use:** only the **MOT subset**. For each of 10 matches it contains one **4-minute clip** (25 frames per second, about 6,000 frames, about 4096 × 1080 px) and one **MOT annotation file** with a box around each of the **22 players** in every frame. A typical player box is only about 20 × 45 px.

### 1.2 Five kinds of files (do not confuse them)

| Kind | Example | What it is | In Git? |
|---|---|---|---|
| **Video** | `mot/clips/117093.mp4` | Input images (decoded into frames) | **Never** |
| **MOT labels** | `mot/117093.txt` | **Human-annotated** ground-truth player boxes for every frame. The only labels we train and evaluate on | **Never** (dataset file) |
| **Calibration** | `raw/117093/117093_keypoints.json`, `_camera_intrinsics.npz` | Pitch landmarks and camera parameters; only needed for Phase 3 pitch mapping | **Never** |
| **Model weights** | `yolov8n.pt` | Pretrained network parameters (COCO, Ultralytics, AGPL-3.0) for Pipeline B | **Never** |
| **Generated outputs** | `outputs/detections/...csv`, `outputs/metrics/...json`, `.npz` patch sets | What our code produces | **Never** (git-ignored; recreated from code) |

Only code, configs, documentation, notebooks (without outputs) and the tiny synthetic fixtures in `tests/fixtures/` are committed.

### 1.3 The raw MOT format (10 columns, verified on the real files)

```
0,1,2033.86,193.80,21.24,53.82,-1,-1,-1,-1
│ │   │       │      │     │    └──┴──┴──┴─ columns 7–10: in the MOTChallenge format these are a confidence/"consider"
│ │   │       │      │     │                flag, a class, a visibility value and an unused field. In SoccerTrack v2
│ │   │       │      │     │                they are -1 in EVERY row: no information; we do not use them
│ │   │       │      │     └─ h: box height in pixels
│ │   │       │      └─ w: box width in pixels
│ │   │       └─ y: top edge of the box (pixels from the TOP of the image)
│ │   └─ x: left edge of the box (pixels from the LEFT of the image)
│ └─ id: player identity, 1–22, constant over the clip
└─ frame: 0-based frame index
```

This real first line of `117093.txt` says: in frame 0, player 1 occupies x = 2033.86…2055.10 and y = 193.80…247.62 (full-frame pixels).

**Frame indexing:**
- Frames are **0-based**. `frame = k` is the **k-th image you get when decoding the video from the start, one frame after another** (k = 0, 1, 2, …).
- Clip 117093 has 5,976 frames, so its labels use frames 0 … 5975.
- Never jump to a position by time: the MP4 container of 117093 starts at 0.979 s, so time-based seeking gives the wrong image.
- The official SoccerTrack documentation says MOT frames are 1-based, but the released files we checked are 0-based. Our contracts follow the files.

**Coordinates:**
- All boxes are in **full-frame pixels**, as top-left `x, y` plus width/height (`xywh`).
- If code works on a downscaled image or on a tile, it must convert boxes back to full-frame coordinates before writing them.

### 1.4 From one frame and its labels to detector and evaluator inputs

```
117093.mp4 ──sequential decode──► frame 0: BGR uint8 array, shape (1080, 4096, 3)     (OpenCV order: Blue, Green, Red)
117093.txt ──read + clean──────► Track table rows for frame 0, e.g.
                                  frame,track_id,x,y,w,h,score
                                  0,1,2033.86,193.80,21.24,53.82,1.0000             (GT: score is always 1.0)

Pipeline A (Member 1): crop each candidate box (+10 % context) → resize → RGB uint8 patch (64, 32, 3) → classifier score
Pipeline B (Member 2): cut the BGR frame into tiles → YOLOv8n → boxes shifted back to full-frame coordinates
Both write a Detection table (no identities):   frame,x,y,w,h,score   e.g.  0,2034.10,194.00,21.00,54.00,0.9100
Trackers turn detections into a predicted Track table (adds track_id):  0,7,2034.10,194.00,21.00,54.00,0.9100
Evaluator (Member 3): for frame 0, IoU between the predicted box and the GT box ≈ 0.95 ≥ 0.5 → true positive.
  For tracking, the identities across frames are compared (GT id 1 ↔ predicted id 7 must stay paired).
```

The three tables are defined precisely in [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md):

| Table | Has `track_id`? | Produced by |
|---|---|---|
| **Ground-truth Track table** | yes (1–22) | MOT labels via `data_io` |
| **Detection table** | no | Detectors |
| **Predicted Track table** | yes (tracker IDs) | Trackers |

**Colour order:** frames are **BGR** (as OpenCV reads them); classifier patches are **RGB**. The conversion happens once, in `patches.crop_patch`.

### 1.5 What we do NOT mix with the MOT labels
- **SoccerTrack GSR files** (`gsr/`, 2.7–2.9 GB per half): their boxes are **generated from pitch positions** with a position-dependent size, not drawn around players. They are not ground truth for a detector and they cover full 45-minute halves.
- **FIFA World Cup 2022 tracking data** (present on the lead machine from an older project): a different competition and provider, with no video and no boxes, different sampling rates and coordinate axes, and an unverified licence.
- **The older project's generated results** on the lead machine: reference numbers only, never our results.

Combining any of these with the MOT labels silently would make our training and evaluation invalid. They are excluded unless the whole team decides otherwise.

### 1.6 Split and leakage rules

| Split | Matches | Use |
|---|---|---|
| Train | 118575, 118576, 118577, 118578, 128058 (117092 optional, off by default) | Fitting: patches, classifiers, size model, normalisation, Phase 3 fine-tuning |
| Validation | 117093 | All choices: thresholds, parameters, epochs, failure analysis |
| Test (held out) | 128057, 132831 (the official SoccerTrack Challenge test pair) | Only in the logged frozen test run, once per phase |

**Why split by match?**
- Frames of the same match are nearly identical: same teams, camera, lighting and players.
- Splitting frames of one match into train and test would let a model "recognise" the test images, so the result would be meaningless.

**Rules:**
- Never fit anything on validation or test.
- Never choose anything on test.
- Never open test clips or labels outside the frozen run (`--frozen-test`; `outputs/test_log.md`).

---

## 2. Three data operating modes

Nobody needs the whole dataset (about 145 GB; never download it). Most first milestones need **no data at all**.

### Mode 1: Fixtures only (default for everyone at the start)
- **Needs:** the repository and the Python environment. No dataset, no weights, no private paths, no GPU, no `configs/data_sources.yaml`.
- **Data used:** `tests/fixtures/*` plus synthetic frames and patches generated in memory by `tests/conftest.py`.
- **Enables:**
  - all contract and unit tests (`pytest`);
  - Member 1 M1 parsing and cleaning logic (plus a tiny synthetic video written by the test itself), M3–M6 logic tests;
  - Member 2 M1 (tiling, mapping, NMS), M4 ByteTrack tests on fixtures (needs the ultralytics package, not weights);
  - Member 3 M0–M4 completely;
  - Member 4 M0 (config, data-check script on temporary fake files, download dry run), M1 (EDA figures and overlays from fixtures), notebook and report layout.

### Mode 2: Member-specific local data (only after team approval)
- A member downloads **the minimum official files their role needs** to their own computer:
  - with their **own** Hugging Face account that has accepted the dataset terms, and their **own** token (never shared);
  - **only** the files listed for their role in Section 3, at the pinned revision;
  - stored **outside Git**, normally in their clone's git-ignored `data/raw/soccertrack-v2/` (or any folder on a drive with enough space);
  - referenced in their own git-ignored `configs/data_sources.yaml`.
- Caches and temporary files go on a drive with free space. Use whatever drive letter exists on **your** computer; D: and F: are only examples (Section 4.1).
- Download command (explicit files only, never the whole repository):
  ```powershell
  hf auth login                                   # newer huggingface_hub; older versions: huggingface-cli login
  hf download atomscott/soccertrack-v2 mot/117093.txt mot/clips/117093.mp4 `
     --repo-type dataset --revision 325bd4c57ee8e135ab652f3921d274526ada496b `
     --local-dir data/raw/soccertrack-v2
  # older huggingface_hub versions: huggingface-cli download <same arguments>
  ```
  Check which command your installed version provides with `hf --help` or `huggingface-cli --help`. The `huggingface_hub` version is not pinned yet.
- Verify every file by byte size (Section 5) and, once implemented, with `python scripts/check_data.py`.

### Mode 3: Official experiments on the lead machine
- The lead machine holds the existing, verified train/validation files and the COCO YOLO weights on `F:\Football-Fatigue-Project` (**read-only**; that old project is never modified).
- The two missing test matches are downloaded there once, after approval (≈ 0.91 GB).
- **All reported numbers** (validation and frozen test) are produced on the lead machine from the integrated code on `main`.
- Other members do not need to own or redistribute the dataset. They contribute **code** through GitHub pull requests; after a merge, the lead machine pulls `main` and runs the runners (`INTEGRATION_PLAN.md` §4 and §5.1).
- Generated artefacts stay in the lead machine's git-ignored `outputs/`. Small results that others need for figures or the report (Metrics JSON, tables, a few overlay PNGs) may be shared through the team's private shared drive. They are never committed, except curated report figures under `docs/phase2/figures/`.

---

## 3. Your data plan by member

| Member | Needed NOW (first milestone) | Source | Local destination / configuration | How to verify | Needed LATER |
|---|---|---|---|---|---|
| **1** | Nothing beyond fixtures for M1 logic tests. Real-file check: one MOT txt (+ the val clip for decoding) | Mode 3 (lead machine), or Mode 2: `mot/117093.txt` (6,221,412 B) + `mot/clips/117093.mp4` (304,980,773 B) | `data/raw/soccertrack-v2/mot/…`; entry `"117093"` in `configs/data_sources.yaml` | Byte sizes; `pytest -m data tests/test_data_io.py` (decoded frames = 5,976 = max frame + 1) | Train txt + clips (5 matches, ≈ 1.47 GB) for patches and training, or run on the lead machine |
| **2** | Nothing (M1 is pure NumPy on synthetic boxes) | — | — | `pytest tests/test_detector_yolo.py` | M2/M3: val clip + `yolov8n.pt` (Section 5.4); Phase 3: train clips + `yolov8s.pt` |
| **3** | Nothing (fixtures contain GT, predictions and expected answers) | — | — | `pytest tests/test_contracts.py tests/test_metrics.py tests/test_evaluate.py` | Real GT txt files (~6 MB each) and producers' CSV/JSON outputs; or evaluate on the lead machine |
| **4** | Nothing (template config, temporary fake files, EDA fixture, synthetic frames) | — | — | `pytest tests/test_config.py tests/test_visualize.py` | M2: val clip for real overlays; Phase 3: calibration files (`raw/<id>/`, KB-sized) |

---

## 4. Python environment (after approval)

### 4.1 Put caches and temporary files on a drive with space
On the lead machine C: has only ~2 GB free, and pip and Hugging Face would fill it. Pick a drive that exists on **your** computer and has at least ~8 GB free:

```powershell
$Drive = "D:"                                      # change to a drive that exists on YOUR computer (e.g. "C:" if it has space)
New-Item -ItemType Directory -Force "$Drive\pip_cache", "$Drive\tmp", "$Drive\hf_cache" | Out-Null
$env:PIP_CACHE_DIR = "$Drive\pip_cache"
$env:TMP  = "$Drive\tmp"
$env:TEMP = "$Drive\tmp"
$env:HF_HOME = "$Drive\hf_cache"
```

To keep these permanently for your account:
```powershell
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "$Drive\pip_cache", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "$Drive\hf_cache", "User")
```

### 4.2 Create the environment
Requirements:
- Python **3.12** (`py -3.12 --version`) and Git for Windows.
- Optional: NVIDIA GPU with a recent driver (lead machine: GTX 1650 Ti, 4 GB). Approach A runs on CPU.

```powershell
cd <your clone>                                    # e.g. D:\pattern_prj
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1                       # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt                    # pinned; PyTorch from the cu126 index in the file
pip install -e . --no-deps                         # makes `import soccertrack` work
```

In VS Code, select `.venv\Scripts\python.exe` (**Ctrl+Shift+P → Python: Select Interpreter**).

`huggingface_hub`, `ipykernel` and `nbconvert` are not pinned yet. Member 4 pins them after the first approved install.

### 4.3 Verify
```powershell
pytest                                             # works immediately (Mode 1). Most tests are SKIPPED placeholders
pytest -rs                                         # lists the skipped acceptance tests and their milestones
pytest tests/test_environment.py                   # all packages import
pytest -m gpu tests/test_environment.py            # CUDA visible (GPU machines only)
```

**A passing run with skips does not mean anything is implemented.** A milestone is complete only when its skipped tests are replaced by real, passing tests.

---

## 5. Data and weights: locations, sizes and access

### 5.1 Lead machine (Mode 3): existing train/val files, referenced in place, read-only
These files were verified against v1.2 (`docs/planning/transfer_manifest.md`). Never copy, move, rename or write to them.

| Match | Split | MOT txt (bytes) | Clip (bytes) |
|---|---|---|---|
| 118575 | train | `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\118575.txt` (6,122,283) | `…\soccertrack-v2\mot\clips\118575.mp4` (287,015,664) |
| 118576 | train | `…\mot\118576.txt` (6,231,376) | `…\mot\clips\118576.mp4` (298,409,129) |
| 118577 | train | `…\mot\118577.txt` (6,250,160) | `…\mot\clips\118577.mp4` (270,151,489) |
| 118578 | train | `…\mot\118578.txt` (4,540,066) | `…\mot\clips\118578.mp4` (294,368,084) |
| 128058 | train | `…\mot\128058.txt` (6,208,255) | `…\mot\clips\128058.mp4` (291,440,428) |
| 117092 | train (optional) | `F:\Football-Fatigue-Project\data\raw\soccertrack_v2\mot\117092.txt` (6,245,767) | `…\raw\soccertrack_v2\mot\117092.mp4` (587,873,512) |
| **117093** | **val** | `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\117093.txt` (6,221,412) | **`F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4`** (304,980,773; unusual folder) |

- `…\soccertrack-v2` = `F:\Football-Fatigue-Project\data\external\soccertrack-v2`.
- On Hugging Face, every clip (including 117093) is at `mot/clips/<id>.mp4` and every label file at `mot/<id>.txt`. Byte sizes are identical.
- **Calibration (Phase 3):** `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\<id>\` for 117092, 117093, 118575, 118576.

### 5.2 Test files: missing; download once on the lead machine after approval (≈ 0.91 GB)

| File | Bytes |
|---|---|
| `mot/README.md` | 2,589 |
| `mot/128057.txt`, `mot/clips/128057.mp4` | 6,218,634 + 291,142,915 |
| `mot/132831.txt`, `mot/clips/132831.mp4` | 6,234,307 + 604,313,569 |
| `raw/128057/*`, `raw/132831/*` (small calibration files) | ~12 KB |
| Optional, separate approval: `mot/132877.txt`, `mot/clips/132877.mp4` | 6,208,062 + 302,437,889 |

- Destination: `<repo>\data\raw\soccertrack-v2\` (git-ignored), keeping the dataset's folders.
- Tool: `scripts/download_test_matches.py`. It does not work yet; Member 4 implements it in M0. The real download runs once, by the person approved to run it, with their own login.
- The corrected 132831 keypoints (`data_corrections/132831_keypoints.json`, official GitHub repository) are Phase 3 only and need separate approval.

### 5.3 Access rules (all modes)
1. Every person who downloads anything uses their **own** Hugging Face account that has accepted the terms at https://huggingface.co/datasets/atomscott/soccertrack-v2, and their **own read token**.
2. Log in with `hf auth login` (newer versions) or `huggingface-cli login` (older). The token is stored under `HF_HOME` in your user profile, never in the repository, a notebook, a config file or a chat.
3. Download only explicitly named files at the pinned revision. Never snapshot the repository (~145 GB).
4. Never share tokens, datasets or weights through Git.

### 5.4 COCO-pretrained YOLO weights on a machine without F:
The weights on the lead machine's F: drive are **not** available on other computers automatically.

Reference files (licence AGPL-3.0). Every copy used by the team must match these hashes:

| Key in `data_sources.yaml` | Lead-machine file | Bytes | SHA-256 |
|---|---|---|---|
| `yolov8n_coco` | `F:\Football-Fatigue-Project\models\yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `yolov8s_coco` (Phase 3) | `F:\Football-Fatigue-Project\models\yolov8s.pt` | 22,588,772 | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |

These hashes were computed from the local files on 2026-11-19. They are not checked against an official Ultralytics checksum list.

Options, in order of preference:
1. **No weights at all.** CPU tests mock `ultralytics.YOLO`. This is enough for Member 2's M1 and the non-GPU parts of M3/M4.
2. **Team copy (needs team approval).** The lead-machine owner shares `yolov8n.pt` (6.5 MB) through the team's private shared drive. Store it **outside the repository** (e.g. `<Drive>\models\yolov8n.pt`), verify it, then reference it in your git-ignored config:
   ```powershell
   (Get-FileHash "$Drive\models\yolov8n.pt" -Algorithm SHA256).Hash.ToLower()   # must equal the SHA-256 above
   ```
   ```yaml
   weights:
     yolov8n_coco: D:/models/yolov8n.pt      # your own path
   ```
3. **Official Ultralytics release asset (needs team approval).** Download `yolov8n.pt` manually in the browser from the official Ultralytics assets releases page (https://github.com/ultralytics/assets/releases) into the same folder outside the repository, then verify the hash. If the hash differs, do not use it; report it, because results would not be comparable.

**Never** commit weights (`*.pt` is git-ignored) or put them inside the repository. **Never** call `YOLO("yolov8n.pt")` with a bare name or a non-existent path: Ultralytics then downloads the file automatically. `detector_yolo.py` must check that the configured file exists first.

---

## 6. Local path configuration (`configs/data_sources.yaml`)

The file is **git-ignored** and machine-specific. Create it only when you leave Mode 1:

```powershell
Copy-Item configs\data_sources.example.yaml configs\data_sources.yaml
code configs\data_sources.yaml
```

- **Lead machine (Mode 3):** replace `<ST_LOCAL>` with `F:/Football-Fatigue-Project` and `<REPO>` with the clone path (e.g. `D:/pattern_prj`). Use forward slashes.
- **Mode 2:** replace `<REPO>` with your clone path. For every match you downloaded, change the entry to the Hugging Face layout, e.g.
  ```yaml
  "117093": {mot: C:/work/pattern_prj/data/raw/soccertrack-v2/mot/117093.txt, clip: C:/work/pattern_prj/data/raw/soccertrack-v2/mot/clips/117093.mp4, calibration_dir: null}
  ```
  Delete or ignore entries you do not have. Only data-marked tests and real runs read this file.
- `config.load_data_sources()` rejects unfilled `<ST_LOCAL>` / `<REPO>` placeholders. This works once Member 4 has implemented M0.

---

## 7. Running tests

| Purpose | Command | Works |
|---|---|---|
| Contract and unit tests (before every push) | `pytest` | Immediately after installation (Mode 1) |
| Show remaining placeholders | `pytest -rs` | Immediately |
| One file, verbose | `pytest tests/test_metrics.py -v` | Immediately (mostly skips until implemented) |
| Real-data tests | `pytest -m data` | Mode 2/3 after `data_io` (Member 1, M1) and `config` (Member 4, M0) are implemented |
| GPU tests | `pytest -m gpu` | GPU machine; YOLO test after Member 2's M2 |
| Data verification | `python scripts/check_data.py --split all` | After Member 4's M0; Mode 2/3 |
| Runner help | `python scripts/run_yolo.py --help` (any runner) | Immediately (argument parsing only) |
| Runner commands | `python scripts/run_yolo.py detect …` | Only after the owner implements them; until then they raise `NotImplementedError` |
| Pipeline smoke test | `INTEGRATION_PLAN.md` §5.1 | Lead machine, after the foundation and the relevant milestones are merged |

---

## 8. Common setup failures

| Symptom | Cause / remedy |
|---|---|
| `No space left on device` / pip fails unpacking torch | Cache or TEMP on a full drive. Set them per Section 4.1 and retry |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No matching distribution found for torch==2.9.1+cu126` | The `--extra-index-url` line was removed, or Python is not 3.12 |
| `torch.cuda.is_available()` is `False` | CPU wheel installed or driver too old. Check `nvidia-smi`; reinstall torch from the cu126 index |
| `ModuleNotFoundError: soccertrack` | Run `pip install -e . --no-deps` inside the activated `.venv` |
| `NotImplementedError: TODO(Member n, Mk)` | Expected: that function is still a skeleton owned by Member n. Use fixtures or a mock until it is merged |
| `401 Unauthorized` / `GatedRepoError` | Terms not accepted for your account, or not logged in |
| Ultralytics starts downloading `yolov8n.pt` | A bare name or a missing path was passed. Use an existing absolute path (Section 5.4) |
| `FileNotFoundError` for an `F:\` path | You are not on the lead machine. Use Mode 1 or Mode 2 paths in your own `data_sources.yaml` |
| Frame-count mismatch | Decode sequentially; never seek by time. Report to Member 1 |
| TrackEval errors about NumPy | Use the pinned `trackeval==1.3.0` + `numpy==2.5.4` |
