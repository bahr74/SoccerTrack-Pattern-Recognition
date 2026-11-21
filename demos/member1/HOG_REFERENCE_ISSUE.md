# OpenCV HOG reference issue — verified 2026-11-20

The current project environment reports OpenCV **5.0.0**, NumPy **2.5.4**.
`hasattr(cv2, 'HOGDescriptor')` is false, and `cv2.xobjdetect` is unavailable.
The exact failing check is `tests/test_patches_features.py:56` (`test_05`).
It remains a failure, not a skip, substitute reference or weakened assertion.

OpenCV's [5.0 migration notes](https://github.com/opencv/opencv/wiki/OpenCV-5)
state that Haar/HOG detectors moved to `opencv_contrib` / `xobjdetect`. This
explains the absent API in this main-module build. The [4.x HOG documentation](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html)
documents the required reference constructor. API absence prevents comparison;
it does **not** establish numerical correctness of the NumPy implementation.

## Exact recommended next step — not executed

For dependency-owner approval, create a **separate reference-only Python 3.12
virtual environment** with `opencv-python==4.12.0.88` and `numpy==2.2.6`, plus
compatible test dependencies. Do not change this project's working environment
or shared requirement pins. The [4.12.0.88 release metadata](https://pypi.org/pypi/opencv-python/4.12.0.88/json)
requires NumPy >=2,<2.3 on Python >=3.9, so installing it directly into the current
NumPy 2.5.4 environment would introduce a compatibility change. Check dependency
resolution and `pip check` in the isolated environment before running the reference
test. No packages/environment were created or installed in this task.

Only one OpenCV wheel flavor may own `cv2` in an environment; the [official wheel
packaging instructions](https://pypi.org/project/opencv-python/4.12.0.88/) explicitly
warn against co-installing main/contrib/headless variants. An alternative
contrib-5 reference environment would require verifying its Python binding and
namespace and explicitly reviewing the test adapter; it is not a guaranteed fix
for a test that calls `cv2.HOGDescriptor` directly.

Once available, rerun `test_05` on the same RGB/grayscale inputs and correct
block/cell x/y ordering. The current tolerance is minimum cosine similarity >0.75,
not exact parity. NumPy HOG omits reference Gaussian block weighting and may differ
at spatial boundaries. If the comparison fails, investigate those numerical
choices; do not assume an environment change resolves algorithmic differences.
The 756-D geometry and structural tests pass independently, but reference parity
remains **UNVERIFIED**. No checkpoint was retrained with a different descriptor.

## Corrective-pass environment inspection (2026-11-21)

No compatible local reference was found among the available interpreters checked:

- Project `.venv`: OpenCV 5.0.0; no `HOGDescriptor`.
- Existing `F:\Football-Fatigue-Project\.venv`: OpenCV 5.0.0; no `HOGDescriptor` (read-only import check).
- Base Python 3.12: NumPy/OpenCV/pytest are not installed.
- The launcher advertises `C:\Python314\python.exe`, but that executable is absent.
- No interpreter exists at `D:\SoccerTrack-v2\.venv\Scripts\python.exe`.

This is an inspection of known local environments, not a claim that every disk
location was exhaustively searched. No package or model was installed/downloaded.
The required numerical comparison is still BLOCKED.

### Proposed isolated commands — approval required, NOT EXECUTED

The separate reference environment only needs the dependencies imported by the
specific test module; it does not need Torch, model training or real match files.
The proposed Python 3.12 pins below keep OpenCV 4.12 within its NumPy >=2,<2.3
requirement. SciPy/pandas/PyYAML are needed by the test module's imports.

```powershell
# Proposal only: run after dependency-owner approval.
& 'C:\Users\DELL\AppData\Local\Programs\Python\Python312\python.exe' -m venv outputs/member1_hog_reference
.\outputs\member1_hog_reference\Scripts\python.exe -m pip install numpy==2.2.6 opencv-python==4.12.0.88 scipy==1.15.3 pandas==2.2.3 PyYAML==6.0.2 pytest==9.1.1
.\outputs\member1_hog_reference\Scripts\python.exe -m pip check
.\outputs\member1_hog_reference\Scripts\python.exe -c "import cv2,numpy; print(cv2.__version__,numpy.__version__); assert hasattr(cv2,'HOGDescriptor')"
.\outputs\member1_hog_reference\Scripts\python.exe -m pytest tests/test_patches_features.py::test_05 -v --basetemp outputs/member1_hog_reference/test_tmp -o cache_dir=outputs/member1_hog_reference/pytest_cache
```

Run from the repository root; pytest already supplies `src` on the import path,
so no editable project install is needed for this reference-only test. Do not
co-install multiple cv2 wheel flavors or change the current project environment.
The proposed environment has not been installed or execution-tested here.

Closure evidence must include interpreter/package versions, a successful `pip check`,
reference constructor availability, per-example cosine similarities after the
specified ordering conversion, the existing >0.75 assertion passing, zero-image
behavior, and an unchanged test/source hash. If numerical comparison fails,
investigate Gaussian block weighting, boundary voting/gradient treatment and
orientation/order conventions before changing the descriptor or tolerance.
An API check alone does not close the numerical finding.
