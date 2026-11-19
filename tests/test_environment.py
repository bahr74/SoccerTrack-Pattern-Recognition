"""Environment smoke test. Owner: Member 4. Run after installing requirements.txt (`pytest -m gpu` for the CUDA check)."""

from __future__ import annotations

import importlib

import pytest

PACKAGES = ["numpy", "scipy", "pandas", "sklearn", "cv2", "yaml", "matplotlib", "torch", "torchvision",
            "ultralytics", "trackeval", "pycocotools", "lap"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    importlib.import_module(name)


def test_package_is_importable():
    import soccertrack

    assert soccertrack.__version__


@pytest.mark.gpu
def test_cuda_visible():
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA not visible: check the cu126 wheel and the NVIDIA driver"
