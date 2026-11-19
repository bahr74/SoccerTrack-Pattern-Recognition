# Transfer Manifest — Football Fatigue Project → `D:\pattern_prj`

**Status:** PROPOSAL ONLY. Nothing has been copied, moved, linked, or modified. Every action below requires explicit approval.
**Audit date:** 2026-11-19. **Method:** read-only directory listing, file sizes, git-blob SHA-1 / SHA-256 hashing, and reading small text/JSON/Markdown files. No large file was loaded into memory; the original project was not modified.

## 1. Rules applied

1. Large files (videos, model weights, datasets) are **referenced in place**, never duplicated. They are referenced through one config file (`configs/data_sources.yaml`, to be created by Codex after approval) that maps each resource to its absolute path.
2. Only small, genuinely reusable metadata may be copied, and only after approval.
3. Nothing from the source project is moved, deleted, or edited.
4. Never transferred: `.git` history, `.venv`, `.pytest_cache`, `__pycache__`, `.claude`, Hugging Face `.cache` folders (including the 0-byte `.incomplete` file), secrets/tokens, generated outputs under `data/processed/` (~35 GB), demo archives.
5. Provenance and licence are recorded for every reused item.

## 2. Locations searched

| Location | Result |
|---|---|
| `F:\Football-Fatigue-Project` | **Found.** Git repo, 43 GB total (5.9 GB `data/raw`, 2.3 GB `data/external`, 35 GB `data/processed`, 129 MB `models`). Contains SoccerTrack v2 files, COCO YOLO weights, a football YOLO checkpoint, FIFA WC2022 PFF data, Wadi Degla videos, documentation. |
| `F:\Football-Fatigue-Video` | **Found.** Git worktree of the same repository (`.git` is a 74-byte pointer file). Only derived outputs (515 MB in `data/processed/video/soccertrack-{117092,117093,118575,118576}`). No raw data. Not needed. |
| `D:\Downloads\soccertrack-v2-20261018T225433Z-1-001` | **Empty folder (0 bytes).** Nothing to reuse. |
| `C:\Users\DELL\Downloads\MePhy Dataset\...\Mental Fatigue Inducing Test` | Found by name only (a physiological mental-fatigue dataset). **Unrelated** to SoccerTrack players; cannot be linked to any tracked player. Not inspected further and not recommended. |
| `C:\Users\DELL\.claude\projects\F--Football-Fatigue-*`, `C:\Users\DELL\.codex\memories\...` | Assistant session/memory folders, not data. Not used. |
| D:, F:, `C:\Users\DELL` | Searched to depth 5 for `*fatigue*`, `*soccertrack*`, `*football*`, `*soccer*`, `*fifa*`, `*worldcup*`, `*pff*`. No inaccessible locations were encountered. Other parts of C: (outside the user profile) were not searched. |

## 3. SoccerTrack v2 files found — integrity verification

All found SoccerTrack files belong to HF release **v1.2** (commit `325bd4c57ee8e135ab652f3921d274526ada496b`, recorded in the HF cache metadata and in the source project's download manifest; this is the current HF revision).

| HF path | Local path (F:\Football-Fatigue-Project\…) | Bytes | Verification |
|---|---|---|---|
| `mot/117092.txt` | `data\raw\soccertrack_v2\mot\117092.txt` | 6,245,767 | ✅ git-blob SHA-1 equals HF listing |
| `mot/117093.txt` | `data\external\soccertrack-v2\mot\117093.txt` | 6,221,412 | ✅ git-blob SHA-1 equals HF listing |
| `mot/118575.txt` | `data\external\soccertrack-v2\mot\118575.txt` | 6,122,283 | ✅ |
| `mot/118576.txt` | `data\external\soccertrack-v2\mot\118576.txt` | 6,231,376 | ✅ |
| `mot/118577.txt` | `data\external\soccertrack-v2\mot\118577.txt` | 6,250,160 | ✅ |
| `mot/118578.txt` | `data\external\soccertrack-v2\mot\118578.txt` | 4,540,066 | ✅ |
| `mot/128058.txt` | `data\external\soccertrack-v2\mot\128058.txt` | 6,208,255 | ✅ |
| `mot/clips/117092.mp4` | `data\raw\soccertrack_v2\mot\117092.mp4` | 587,873,512 | ✅ size equals HF; ⚠️ SHA-256 `a6e2d315…` is recorded only by the source project's own run manifests (not independently confirmed against HF) |
| `mot/clips/117093.mp4` | `data\external\soccertrack-v2\raw\117093\117093.mp4` (**note: unusual folder**) | 304,980,773 | ✅ SHA-256 `dfe2260a…` equals the HF ETag stored in the download metadata and in `docs/vision_module.md` |
| `mot/clips/118575.mp4` | `data\external\soccertrack-v2\mot\clips\118575.mp4` | 287,015,664 | ✅ SHA-256 equals pinned HF manifest (`m62-plan-v1/download_manifest.json`) |
| `mot/clips/118576.mp4` | `…\mot\clips\118576.mp4` | 298,409,129 | ✅ |
| `mot/clips/118577.mp4` | `…\mot\clips\118577.mp4` | 270,151,489 | ✅ |
| `mot/clips/118578.mp4` | `…\mot\clips\118578.mp4` | 294,368,084 | ✅ |
| `mot/clips/128058.mp4` | `…\mot\clips\128058.mp4` | 291,440,428 | ✅ |
| `bas/117092/117092_12_class_events.json` | `data\raw\soccertrack_v2\bas\117092\117092_12_class_events.json` | 443,655 | ✅ git-blob SHA-1 |
| `raw/117093/117093_keypoints.json` | `data\external\soccertrack-v2\raw\117093\` | 3,852 | ✅ git-blob SHA-1 |
| `raw/117093/117093_homography.npy`, `_camera_intrinsics.npz`, `_padding_info.csv` | same folder | 200 / 1,996 / 154 | ✅ size; SHA-256 in `docs/vision_module.md`; padding CSV git-blob ✅ |
| `raw/117092/117092_keypoints.json`, `_camera_intrinsics.npz` | `data\external\soccertrack-v2\raw\117092\` | 3,624 / 1,996 | ✅ keypoints git-blob; npz size |
| `raw/118575/118575_keypoints.json`, `_camera_intrinsics.npz` | `…\raw\118575\` | 3,604 / 1,996 | ✅ / size |
| `raw/118576/118576_keypoints.json`, `_camera_intrinsics.npz` | `…\raw\118576\` | 3,604 / 1,996 | ✅ / size |

Not present locally: `117093_mapx.npy` / `_mapy.npy` (documented in the old project but not on disk now); **anything for 128057, 132831, 132877**; `mot/README.md`; all GSR JSON; all full-half videos.

**Total reusable SoccerTrack data ≈ 2.37 GB** (7 of 10 MOT annotation files + 7 of 10 clips + small calibration files).

## 4. Transfer manifest

Columns: Source Path | File/Resource | Size | Purpose | Compatibility | Recommended Action | Destination or Reference Path.
`FFP` = `F:\Football-Fatigue-Project`. `PRJ` = `D:\pattern_prj`.

### 4.1 SoccerTrack v2 data (reference in place)

| Source Path | File/Resource | Size | Purpose | Compatibility | Recommended Action | Destination or Reference Path |
|---|---|---|---|---|---|---|
| `FFP\data\external\soccertrack-v2\mot\{117093,118575,118576,118577,118578,128058}.txt` | MOT GT (6 matches) | 35.6 MB | Train/val GT, EDA | ✅ Same dataset, v1.2, content-verified. Note: 0-based frames, `conf/class/vis = -1` | **Reference in place** | `configs/data_sources.yaml → mot_txt[<match>]` |
| `FFP\data\raw\soccertrack_v2\mot\117092.txt` | MOT GT (M1) | 6.2 MB | Optional train / EDA | ✅ | **Reference in place** | `mot_txt[117092]` |
| `FFP\data\external\soccertrack-v2\mot\clips\{118575,118576,118577,118578,128058}.mp4` | MOT clips (train) | 1.44 GB | Training frames/patches | ✅ v1.2, hash-verified | **Reference in place** (do not copy) | `clip[<match>]` |
| `FFP\data\external\soccertrack-v2\raw\117093\117093.mp4` | MOT clip (val, M2) | 305 MB | Validation | ✅ hash = HF ETag; stored under `raw\` by the old project, which is unusual but the content is correct | **Reference in place** | `clip[117093]` |
| `FFP\data\raw\soccertrack_v2\mot\117092.mp4` | MOT clip (M1) | 588 MB | Optional train (different geometry, stitch seam) | ✅ size; ⚠️ hash only self-recorded | **Reference in place, optional** | `clip[117092]` |
| `FFP\data\external\soccertrack-v2\raw\{117092,117093,118575,118576}\*_keypoints.json`, `*_camera_intrinsics.npz`, `117093_homography.npy`, `117093_padding_info.csv` | Calibration | ~25 KB | Pitch-mask fallback from keypoints (P2), pitch mapping (P3) | ✅ v1.2 | **Reference in place** (or copy, since they are tiny; reference is enough) | `calib[<match>]` |
| `FFP\data\raw\soccertrack_v2\bas\117092\117092_12_class_events.json` | BAS events (one match) | 0.44 MB | Optional bonus | ✅ | **Not needed for Phase 2**; reference if a bonus requires it | `bas[117092]` |

### 4.2 Model weights (reference in place)

| Source Path | File/Resource | Size | Purpose | Compatibility | Recommended Action | Destination or Reference Path |
|---|---|---|---|---|---|---|
| `FFP\models\yolov8n.pt` | Ultralytics YOLOv8n, COCO | 6,549,796 B | Approach B zero-shot (primary) | ✅ Loaded by ultralytics 8.4.182 in the old project. Licence: Ultralytics AGPL-3.0 (fine for coursework; note it in the report). ⚠️ The download source is not recorded locally; Codex should record its SHA-256 | **Reference in place**; avoids a download | `weights.yolo_coco_n` |
| `FFP\models\yolov8s.pt` | YOLOv8s, COCO | 22,588,772 B | Approach B alternative / fine-tune base (P3) | ✅ same as above | **Reference in place** | `weights.yolo_coco_s` |
| `FFP\models\yolov8m.pt` | YOLOv8m, COCO | 52,136,884 B | Not needed (the old project measured no F1 gain over n/s on 117093) | ✅ | **Do not use** | — |
| `FFP\models\vision\player_detector.pt` | `mobadam/football-player-detection` (YOLO26, classes ball/player/referee/goalkeeper) | 53,140,901 B | Optional "football-pretrained" reference | ⚠️ Apache-2.0 per model card; trained on broadcast footage; **its training data is not documented well enough to rule out SoccerTrack overlap** | **Optional, reference only**; never the primary baseline; label results "third-party football detector" | `weights.football_thirdparty` |

### 4.3 Documentation and metadata (small; copy only if approved)

| Source Path | File/Resource | Size | Purpose | Compatibility | Recommended Action | Destination or Reference Path |
|---|---|---|---|---|---|---|
| `FFP\data\processed\vision_eval\m62-plan-v1\download_manifest.json` | Pinned v1.2 file list with sizes and SHA-256 / git-blob hashes (includes the missing test matches) | small (KB) | Verify the new downloads of 128057, 132831, 132877 | ✅ Derived from the public HF tree at commit `325bd4c…` | **Copy (recommended)**: lets us verify the test downloads without another listing | `PRJ\data\manifests\soccertrack_v1.2_download_manifest.json` (gitignored) |
| `FFP\data\processed\vision_eval\soccertrack-official-repo-audit-v1\SOCCERTRACK_OFFICIAL_REPOSITORY_AUDIT.md` | Prior audit of the official repo | 30.6 KB | Background reading for the discussion | ✅ Our own document | **Reference only** | cite path in report notes |
| `FFP\docs\cv_detection_tracking_report.md` | Prior detection/tracking results on 117093 (P/R/IDF1, failure modes, 0.979 s container offset) | 7.6 KB | Validation-set reference numbers and known failure modes | ✅ Our own work; **117093 = our validation match** | **Reference only**; quote numbers as "prior validation-match results" | — |
| `FFP\docs\vision_module.md` | Checksum table, clip geometry (4096×1080, 25 fps, 5,976 frames), calibration diagnostic (held-out median error 2.61 m) | 31 KB | Provenance and calibration-error evidence | ✅ | **Reference only** | — |
| `FFP\src\football_fatigue\vision\{mot_eval.py, tracking_eval.py, tracker.py, detector.py, pitch_mapping.py, calibration.py}` | Prior implementation (TrackEval wrapper with 0-based handling, tiling, ByteTrack wrapper, calibration) | 8–22 KB each | Design reference for Codex | ⚠️ Tied to the old project's data contracts; much larger than our course scope | **Reference only (do not copy)**; Codex writes small new modules, and may consult these for conventions (frame base, TrackEval config) | — |
| `FFP\.venv` (package versions) | Known-good versions: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, matplotlib 3.11.2, torch 2.9.1+cu126, torchvision 0.24.1+cu126, ultralytics 8.4.182, trackeval 1.3.0, lap 0.5.13, supervision 0.30.9 | — | Pin `requirements.txt` to a combination already shown to work on this machine (including TrackEval with NumPy 2.x) | ✅ | **Do not copy the venv.** Copy only the version numbers into `requirements.txt`. Installing a new venv requires approval | `PRJ\requirements.txt` |

### 4.4 Found but NOT recommended

| Source Path | Resource | Size | Why not |
|---|---|---|---|
| `FFP\data\raw\Tracking Data-*`, `Event Data-*`, `Metadata-*`, `Rosters-*` | **FIFA World Cup 2022 PFF FC** tracking (64 × `jsonl.bz2`), events (3 versions), metadata, rosters | ~5.5 GB | **Different dataset**: no video; 29.97 Hz positions (raw effective ≈ 15 Hz); metres, origin at centre, +y "bottom to top" per PFF spec vs SoccerTrack "+y toward the near touchline". **Licence/redistribution terms unverified** (per the old project's `docs/data_policy.md`). Cannot be combined with SoccerTrack MOT boxes. Possible future use only as an independent sanity check of workload code, after licence verification, with explicit coordinate and sampling-rate conversion |
| `FFP\data\external\wadi-degla\raw\*.mp4` (3 videos) | Local club footage (fixed and moving camera) | 687 MB | No annotations; consent/provenance not documented for course use. Possible Phase 3 qualitative "domain-shift" demo only if the team confirms permission |
| `FFP\data\processed\*` (incl. `vision_eval` 19 GB, `vision` 1.2 GB, `m11-*`) | Generated outputs of the old project | ~35 GB | Derived from different pipelines/settings; would contaminate a clean course evaluation. Not transferred |
| `FFP\wadi-degla-demo-assets.zip` | Demo archive | 299 MB | Unrelated |
| `F:\Football-Fatigue-Video\data\processed\video\*` | Derived video outputs | 515 MB | Derived, not raw |
| `FFP\data\external\soccertrack-v2\.cache\huggingface\…\*.incomplete` | 0-byte partial download | 0 B | Leftover |

## 5. What still has to be downloaded

| Priority | HF path (revision v1.2) | Bytes | Why |
|---|---|---|---|
| P0 | `mot/README.md` | 2,589 | Official note on the MOT subset (gated; never read so far) |
| P0 | `mot/128057.txt` + `mot/clips/128057.mp4` | 6,218,634 + 291,142,915 | **Test M7** |
| P0 | `mot/132831.txt` + `mot/clips/132831.mp4` | 6,234,307 + 604,313,569 | **Test M9** |
| P1 (Phase 3) | `raw/{128057,132831}/*_{keypoints.json,camera_intrinsics.npz,homography.npy,padding_info.csv}` | ≈ 12 KB | Pitch mapping on test matches |
| P1 (Phase 3) | GitHub `data_corrections/132831_keypoints.json` | 2,832 | Corrected test-match calibration |
| Optional | `mot/132877.txt` + `mot/clips/132877.mp4` | 6,208,062 + 302,437,889 | Second official validation match (M10) |

**Minimum additional download: ≈ 0.91 GB** (P0). With the optional second validation match: ≈ 1.22 GB.
Destination (proposed): `D:\pattern_prj\data\raw\soccertrack-v2\` (gitignored), with `HF_HOME=D:\hf_cache` because C: has about 2 GB free. The old project is not used as the download target.
