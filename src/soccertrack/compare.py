"""Model comparison: Metrics JSON -> comparison tables (CSV) and evaluation figures.

Owner: Member 3. Inputs: outputs/metrics/*.json only. Outputs: outputs/tables/*.csv, outputs/figures/eval/*.png.
Member 4 embeds these outputs in the notebook, report and slides; Member 4 does not recompute metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def collect_metrics(metrics_dir: Path) -> "pd.DataFrame":
    """Flatten all Metrics JSON files into one long table (one row per run_id x metric)."""
    raise NotImplementedError("TODO(Member 3, M4)")


def detection_table(long_df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """Approach A vs B detection table (per match + pooled): P, R, F1, AP50, AP50-95, FP/frame, FPS."""
    raise NotImplementedError("TODO(Member 3, M4)")


def tracking_table(long_df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """Detector x tracker table incl. oracle rows: HOTA, DetA, AssA, IDF1, MOTA, IDSW."""
    raise NotImplementedError("TODO(Member 3, M4)")


def reference_table() -> "pd.DataFrame":
    """Literature references with comparability labels (plan Section 3.5): never mixed with our numbers."""
    raise NotImplementedError("TODO(Member 3, M4)")


def plot_confusion_matrix(metrics: dict[str, Any], out_path: Path) -> Path:
    """Confusion-matrix figure for an E1/E2 Metrics JSON."""
    raise NotImplementedError("TODO(Member 3, M4)")


def plot_curves(metrics_list: list[dict[str, Any]], curve: str, out_path: Path) -> Path:
    """Overlay PR or ROC curves of several runs (curve = "pr" | "roc")."""
    raise NotImplementedError("TODO(Member 3, M4)")
