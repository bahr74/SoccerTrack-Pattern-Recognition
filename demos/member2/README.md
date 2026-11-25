# Member 2: tiled YOLOv8n + ByteTrack

Local implementation and real validation are complete for the available boundaries.
**Official E3/E4 execution awaits Member4 infrastructure and final integration.**
Member3 contracts/evaluators are implemented, with independent corrective tests.
The authoritative evidence and completion matrix are in [SCIENTIFIC_HANDOFF.md](SCIENTIFIC_HANDOFF.md).

Use the existing `.venv`, from the repository root. No package installation,
weight download, training, held-out access, or Git mutation is part of these commands.
The isolated runner requires the existing approved F-drive validation 117093 files
and the approved local weights SHA-256. It writes only new D-drive output directories.
No working shared data registry is available on this checkout; the isolated demo's
explicit validation-only paths do not replace production config loading.

## Review saved results first

- Source notebook: `demos/member2/notebooks/member2_yolo_pipeline_experiments.ipynb`
- Run: `outputs/member2_scientific/runs/member2_validation_20261125`
- Executed notebook: `<run>/member2_yolo_pipeline_experiments_executed.ipynb`
- Standalone HTML: `<run>/member2_yolo_pipeline_experiments.html`
- Real video: `<run>/annotated_validation.mp4` (150 verified frames, 25 fps)
- Configurations/hashes/frame IDs: `<run>/manifest.json`, `source_snapshot/`, `artifact_hashes.json`
- Exact test logs: `outputs/member2_scientific/*tests.txt`, `full_pytest_final.txt`

The notebook defaults to saved caches and checks artifact/weight hashes. It does
not start inference. Do not stage generated artifacts or large files.

## Explicit reproduction

Choose a NEW output directory; the runner refuses to overwrite existing runs.

```powershell
.venv/Scripts/python.exe demos/member2/scientific_workflow.py --output outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/runtime_probe.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/build_artifacts.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/build_notebook.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/write_handoff.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
```

The archive in each run preserves the exact science-producing sources. CPU fallback
is implemented, but CPU-only full experiments were not timed. Future workflow runs
capture architecture/environment automatically. They can also be inspected separately:
`python demos/member2/environment.py`. The original executed run retains its original
three-file source snapshot and recorded environment; subsequent reproduction helper
improvements do not relabel or overwrite that historical evidence.

Derivative builders now refuse existing outputs. Architecture/environment and
notebook runtime are run-local; the notebook generator writes a fresh run-local
source notebook and preserves the historical tracked notebook. Handoff generation
writes run-local drafts and uses only that run's test logs. Commands above describe
a future explicitly authorized reproduction, not permission to repeat inference.

For cache reuse after Member4 integration, use `demos/member2/migrate_cache.py`:
`verify_cache` checks the source model/config/code/data hashes and exact coverage;
`migrate` requires the actual nonzero shared config hash and writes a fresh output
tree through shared writers/readers; `replay` consumes those serialized detections.
Keep .05-floor detections, even below the reporting threshold. Recompute E3/E4
after migration; raw historical tracks are not official derivatives.

## Tests

Use a NEW basename for `--basetemp` on reruns, avoiding protected old temporary files.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_detector_yolo.py tests/test_tracker_bytetrack.py tests/test_member2_scientific.py -q --basetemp=outputs/member2_scientific/NEW_unit_tmp -o cache_dir=outputs/member2_scientific/pytest_cache
.venv/Scripts/python.exe -m pytest -m gpu tests/test_detector_yolo.py -q -o cache_dir=outputs/member2_scientific/pytest_cache
.venv/Scripts/python.exe -m pytest -m data tests/test_member2_scientific.py -q -o cache_dir=outputs/member2_scientific/pytest_cache
.venv/Scripts/python.exe -m pytest -q -rs --basetemp=outputs/member2_scientific/NEW_full_tmp -o cache_dir=outputs/member2_scientific/pytest_cache --tb=short
.venv/Scripts/python.exe demos/member2/audit.py
```

The final command verifies 7,081 pre-existing protected source/artifact hashes.
The known Member 1 HOG reference failure remains unchanged. Conditional real-data
skips require approved local files/CUDA; they are not implementation placeholders.

## Official commands after integration

```powershell
.venv/Scripts/python.exe scripts/run_yolo.py detect --detector yolov8n_coco --split val --match 117093 --frameset eval10
.venv/Scripts/python.exe scripts/run_yolo.py detect --detector yolov8n_coco --split val --match 117093 --frameset win60s
.venv/Scripts/python.exe scripts/run_yolo.py track --detector yolov8n_coco --split val --match 117093 --frameset win60s
```

Currently these stop at shared config loading. Test-split guards are retained,
`eval10` tracking is rejected, and debug `--max-frames` outputs are isolated under
`outputs/debug/`. Fine-tuning and frozen M6 are outside current authorization.

## Manual staging scope (not executed)

Review only these paths. `demos/member2/` contains small Python/Markdown/source
notebook files; all executed results, video and caches remain ignored in outputs.

```text
src/soccertrack/detector_yolo.py
src/soccertrack/tracker_bytetrack.py
scripts/run_yolo.py
configs/yolo.yaml
tests/test_detector_yolo.py
tests/test_tracker_bytetrack.py
tests/test_member2_scientific.py
docs/phase2/sections/member_2_yolo.md
demos/member2/
```
