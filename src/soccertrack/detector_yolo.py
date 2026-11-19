"""Approach B detector: native-resolution tiling -> COCO-pretrained YOLOv8n ("person") -> global NMS.

Owner: Member 2. Config: configs/yolo.yaml. Weights: configs/data_sources.yaml -> weights.yolov8n_coco
(read-only reference to the existing local yolov8n.pt; see docs/team/SETUP_AND_DATA.md; never copy or download).
Output (Contracts §2, §4, §7): outputs/detections/<detector>/<split>/<match>__<frameset>.csv + .meta.json
Detector IDs: yolov8n_coco (Phase 2); yolov8n_ft / yolov8s_ft (Phase 3, fine-tuned).
Never downscale the whole 4096 px frame: players are ~20 x 45 px.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def make_tiles(width: int, height: int, tile_w: int, tile_h: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return tiles (x0, y0, x1, y1) covering the whole frame with >= `overlap` px overlap; last tile flush with the edge."""
    raise NotImplementedError("TODO(Member 2, M1)")


def tile_to_frame(boxes_xyxy: "np.ndarray", tile: tuple[int, int, int, int]) -> "np.ndarray":
    """Shift tile-local xyxy boxes into full-frame coordinates."""
    raise NotImplementedError("TODO(Member 2, M1)")


def merge_tile_detections(boxes_xyxy: "np.ndarray", scores: "np.ndarray", iou: float) -> tuple["np.ndarray", "np.ndarray"]:
    """Cross-tile duplicate removal with torchvision.ops.nms; return kept boxes and scores."""
    raise NotImplementedError("TODO(Member 2, M1)")


def apply_pitch_filter(dets: "pd.DataFrame", mask_path: Any) -> tuple["pd.DataFrame", str]:
    """Drop detections whose bottom-centre is outside outputs/masks/<match>_pitch.png.

    If the mask file does not exist, return the input unchanged and the note "pitch_filter=off" (Contracts §10).
    """
    raise NotImplementedError("TODO(Member 2, M3)")


class YoloTiledDetector:
    """Wraps ultralytics.YOLO for tiled inference on one full-resolution BGR frame."""

    def __init__(self, cfg: dict[str, Any], weights_path: Any) -> None:
        raise NotImplementedError("TODO(Member 2, M2)")

    def detect_frame(self, frame_bgr: "np.ndarray", frame_id: int) -> "pd.DataFrame":
        """Return Detection-table rows (full-frame xywh, score >= conf_floor) for one frame."""
        raise NotImplementedError("TODO(Member 2, M2)")


def run_detection(detector_id: str, match_id: str, split: str, frameset: str, cfg: dict[str, Any],
                  sources: dict[str, Any], frozen_test: bool = False) -> "pd.DataFrame":
    """Run tiled YOLO on every frame of `frameset`; write the Detection table + metadata (incl. FPS)."""
    raise NotImplementedError("TODO(Member 2, M3)")


def finetune(cfg: dict[str, Any], sources: dict[str, Any]) -> Any:
    """Phase 3: build TRAIN tiles + YOLO labels from GT, fine-tune, early-stop on VAL tiles, save to outputs/models/."""
    raise NotImplementedError("TODO(Member 2, Phase 3)")
