# Football Player Detection & Tracking — Scratch Pipeline
**Corrective update (2026-11-21):** See [CORRECTIVE_HANDOFF.md](CORRECTIVE_HANDOFF.md) for current verification, prediction parity, audit statuses and the complete staging list. Earlier scientific numbers are preserved.


Member 1's isolated scientific application. The official runner remains
`scripts/run_scratch.py`; it requires Members 3/4's unfinished shared interfaces.
Local diagnostics here are visibly provisional and never enter official folders.

## Start the GUI

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe demos/member1/app.py --port 8765
```

Open http://127.0.0.1:8765/ and click **Load Real Validation Demo**. It reads approved
117093 directly and reuses the cached label-free background. No typed paths are
needed. **Browse Video** opens the browser's file chooser and uploads the selected
file to ignored `outputs/member1_scientific/uploads/` (512 MiB maximum). For the
existing synthetic correctness fixture, select `outputs/member1_demo/synthetic.avi`.
GT comparison is available for the registered real demo, not arbitrary uploads.

Use **Detection** or **Tracking** with rules, HOG + SVM or Small CNN. Checkpoint
choices explicitly separate expanded real training, the earlier real pilot and
synthetic training. Metadata, playback, speed, seek, previous/next frame, save-frame
and annotated export are on the analyzer. Detection display uses score 0.5;
tracking uses configured confirmation/age/gate with score 0.5. Preview is capped
at 600 frames; export is the first 30 frames with local track IDs. Rewind/long seeks
replay sequential frames and can be slow. Processing is offline: background
estimation sees samples across the clip. Playback waits for inference.

The other three views explain ten saved pipeline stages, separate patch and detector
comparisons, and expose actual dataset statistics, errors, figures and downloads.
Explorer images are from the saved experiment, not the analyzer's selected frame.
All artifacts are local. The server binds loopback and protects mutations/uploads
and artifact downloads with a process token. Do not share the token or use
untrusted serialized model checkpoints. Close with Ctrl+C.

## Notebook and reproducibility

Source: `demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb`.
It has the 16 required scientific sections in order, reusable module calls and
training **off** by default. Source outputs are cleared for review. Open it in a
notebook-capable editor and select the project `.venv` Python. JupyterLab and the
Notebook web server are not installed here; no dependency was installed for this task.
The following verified command executes using the already available nbclient,
nbconvert and ipykernel, then writes an executed notebook and standalone HTML:

```powershell
.\.venv\Scripts\python.exe demos/member1/build_notebook.py --execute
```

The printed directory is under `outputs/member1_scientific/notebook_executions/`.
Open its `member1_workflow.html` in a browser to read the executed report.
For an explicit lightweight **synthetic correctness** run without F-drive data:

```powershell
.\.venv\Scripts\python.exe demos/member1/build_notebook.py --execute --smoke
```

Smoke mode generates its own artificial video, trains for one CPU epoch and does
not replace the real-run pointer. Its altered negative mix and same-clip split
are labeled; it supplies no real-data performance evidence.

## Reproduce expanded real experiments

The existing isolated ignored registry is `outputs/member1_scientific/local_sources.json`.
On another approved data machine, register the five train matches and validation
117093 explicitly, then run (no automatic downloads):

```powershell
.\.venv\Scripts\python.exe demos/member1/scientific_workflow.py --register-data 'F:\Football-Fatigue-Project'
.\.venv\Scripts\python.exe demos/member1/scientific_workflow.py --run --epochs 8
.\.venv\Scripts\python.exe demos/member1/supplemental_audit.py
```

Each run creates a unique directory and preserves earlier checkpoints. The latest
completed real run is selected by ignored `outputs/member1_scientific/latest.json`.
Train normalization, calibration folds and size fitting exclude validation.
Validation selects C, CNN epoch and the candidate configuration, so report scores
are development results. No final held-out testing is performed.

## Verification and handoff

```powershell
.\.venv\Scripts\python.exe demos/member1/verify_browser.py
.\.venv\Scripts\python.exe -m pytest -q -rs --basetemp outputs/member1_scientific/pytest_tmp_manual -o cache_dir=outputs/member1_scientific/pytest_cache
```

Use a fresh `--basetemp` directory for each run. The browser check uses installed
Edge with real CDP input events/file selection and an isolated ignored profile;
it installs nothing. It closes its server and browser afterward. Native OS chooser
appearance is not verified. MP4 output uses `mp4v`; sequential OpenCV decoding was
verified, not every browser/player's codec support.

See [SCIENTIFIC_HANDOFF.md](SCIENTIFIC_HANDOFF.md) for measured results, precise
paths, M1–M8 status, blockers, changed files and manual staging guidance, and
[HOG_REFERENCE_ISSUE.md](HOG_REFERENCE_ISSUE.md) for the unresolved reference test.
Never stage outputs, data, models, caches or the local source registry.


## Corrective behavior and provenance

The analyzer resolves the selected pipeline/checkpoint preset before preprocessing.
Configuration, background-pixel identity, video fingerprint, size model and model
SHA-256 identify cached results. The visible provenance panel records frame,
model/checkpoint, configuration and background. Exports carry `provenance.json`
and per-frame detection/track CSVs; saved JPEGs have matching JSON sidecars.
The fixed 0.5 preview threshold is illustrative, not a frozen-test operating point.

Original and GT views perform no classifier inference or association. Detection
performs inference only on the requested frame, without tracking. Tracking uses
bounded detection/row caches and up to eight replay checkpoints; raw image caches
hold at most two images and cap retained originals at 64 MiB. One decoded image
must still fit in memory. Sequential grabs preserve zero-based indexing.

Browser file size/format checks happen before upload. A preflight endpoint and
HTTP 413 rejection cover oversized input. The server drains only a bounded amount
of already-in-flight rejected data; it does not consume arbitrarily large rejected
uploads. Processing/Ready/Error/Cancelled states are visible. Cancel stops requests
and cooperative processing between frames; one codec/inference call cannot be
interrupted instantly. Failed/cancelled views do not retain pending predictions.
Operational UTC JSONL logs are under `outputs/member1_scientific/corrections/runtime/`.

A retrospective supplement, not a rewrite of historical metadata, is selected by
`outputs/member1_scientific/corrections/latest.json`. It records actual background
indices/counts, eligible versus selected frames, derived seeds, patch duplication
and uniquely weighted sensitivity metrics. It also supplies a new aspect-correct
patch contact sheet. No historical arrays, weights, metrics or figures are replaced.

Future scientific runs save `resolved_settings.json`. Validate a recorded plan
without preparing data or training:

```powershell
.\.venv\Scripts\python.exe demos/member1/scientific_workflow.py --replay-settings <resolved_settings.json>
```

`--prepare-only --replay-settings <path>` prepares a new versioned patch output,
without model training or replacing the latest completed-run pointer. Full
training replay is implemented for separately authorized future use; it was not
executed in this corrective pass. It reads saved settings rather than current
YAML, checks source/background consistency and retains the original replacement
sampling policy. Deterministic retraining has not been demonstrated.

Run the corrective suite, including the six approved real-data parity cases:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_member1_corrective.py tests/test_member1_demo.py tests/test_member1_scientific.py -m "data or not data" -q --basetemp outputs/member1_corrective_tmp_manual -o cache_dir=outputs/member1_corrective_pytest_cache
```

Use a fresh temporary directory on subsequent runs. Real-data cases skip explicitly
when approved local evidence is absent; no downloads are performed.
