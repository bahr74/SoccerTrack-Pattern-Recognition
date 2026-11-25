"""Visualisation: overlays, EDA figures, segmentation stage figures and Phase 3 trajectory/heatmap figures.

Owner: Member 4. Consumes contract artefacts only (Track/Detection tables, EDA stats JSON, stage images,
PitchTrack tables). Evaluation figures (confusion matrices, PR/ROC curves) belong to compare.py (Member 3).
Writes to outputs/figures/{eda,overlays,workload}/ ; curated copies go to docs/phase2|3/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _stable_color_bgr(track_id: int) -> tuple[int, int, int]:
    """Return a stable BGR colour for *track_id* using a hash → HSV mapping.

    The same track ID always receives the same colour regardless of frame order.
    """
    import colorsys
    # Use a simple hash spread across the hue wheel; salt with a prime to
    # avoid clustering IDs 1–22 near each other.
    hue = (track_id * 137 % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


# ---------------------------------------------------------------------------
# M1: draw_boxes
# ---------------------------------------------------------------------------

def draw_boxes(
    frame_bgr: "np.ndarray",
    table: "pd.DataFrame",
    frame_id: int,
    color_by: str = "track_id",
    label: bool = True,
) -> "np.ndarray":
    """Draw boxes of one frame from a Track or Detection table onto a BGR image.

    Returns a *new* image (the input is never modified). Colour is stable per
    *track_id* (hash → HSV). Falls back to a fixed green if *color_by* column
    is absent (e.g. Detection tables that have no track_id).

    Parameters
    ----------
    frame_bgr:
        Source image, uint8 BGR, shape (H, W, 3).
    table:
        Track or Detection table (pandas DataFrame) conforming to
        SHARED_CONTRACTS.md § 3 or § 4.
    frame_id:
        0-based frame index; only rows where ``table.frame == frame_id`` are drawn.
    color_by:
        Column name used for colour assignment. If the column is missing, all
        boxes are drawn in green.
    label:
        If True, draw the ID or score as text above the box.
    """
    import cv2  # type: ignore
    import numpy as np

    out = frame_bgr.copy()
    subset = table[table["frame"] == frame_id]

    lw = 2  # line width

    for _, row in subset.iterrows():
        x, y, w, h = int(round(row["x"])), int(round(row["y"])), int(round(row["w"])), int(round(row["h"]))
        x2, y2 = x + w, y + h

        # Colour
        if color_by in row.index:
            color = _stable_color_bgr(int(row[color_by]))
        else:
            color = (0, 200, 0)  # green fallback

        cv2.rectangle(out, (x, y), (x2, y2), color, lw)

        if label:
            if "track_id" in row.index:
                txt = str(int(row["track_id"]))
            elif "score" in row.index:
                txt = f"{row['score']:.2f}"
            else:
                txt = ""
            if txt:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.45
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(txt, font, scale, thickness)
                # Background rectangle for readability
                tx, ty = x, max(0, y - th - baseline - 2)
                cv2.rectangle(out, (tx, ty), (tx + tw, ty + th + baseline + 2), color, -1)
                cv2.putText(out, txt, (tx, ty + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# M1: plot_eda
# ---------------------------------------------------------------------------

def plot_eda(
    stats: dict[str, Any],
    box_sample: "pd.DataFrame | None",
    out_dir: Path,
) -> list[Path]:
    """Produce EDA figures from the ``outputs/eda/stats.json`` structure.

    Figures written (all PNG at ``visualization.figure_dpi`` = 150):

    1. ``boxes_per_match.png``    — bar chart of annotated boxes per match.
    2. ``box_size_hist.png``      — width / height histograms (p05, median, p95).
    3. ``height_vs_row.png``      — box height vs. image row (from box_sample if given).
    4. ``track_length.png``       — track length distribution (min/median/max per match).
    5. ``data_issues.png``        — table of per-match quality issues.

    Parameters
    ----------
    stats:
        Parsed JSON dict from ``outputs/eda/stats.json`` (or the fixture).
    box_sample:
        Optional DataFrame from ``outputs/eda/box_sample.csv`` with columns
        ``match, split, frame, x, y, w, h``.  If *None*, figure 3 is skipped.
    out_dir:
        Target directory; created if it does not exist.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DPI = 150
    matches_data = stats.get("matches", {})
    match_ids = list(matches_data.keys())
    produced: list[Path] = []

    # ---- 1. Boxes per match (bar chart) ------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, len(match_ids) * 0.9), 4), dpi=DPI)
    boxes = [matches_data[m].get("boxes", 0) for m in match_ids]
    splits = [matches_data[m].get("split", "?") for m in match_ids]
    split_colors = {"train": "#2196F3", "val": "#FF9800", "test": "#F44336"}
    colors = [split_colors.get(s, "#9E9E9E") for s in splits]
    bars = ax.bar(match_ids, boxes, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Annotated Boxes per Match", fontsize=12, fontweight="bold")
    ax.set_xlabel("Match ID")
    ax.set_ylabel("Number of Boxes")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.tick_params(axis="x", rotation=30)
    for split_name, col in split_colors.items():
        ax.bar([], [], color=col, label=split_name)
    ax.legend(title="Split", fontsize=8)
    # Totals reference
    totals = stats.get("totals", {})
    if totals:
        ax.annotate(
            f"Total: {totals.get('boxes', '?'):,} boxes / {totals.get('frames', '?'):,} frames",
            xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=8, color="grey",
        )
    fig.tight_layout()
    p = out_dir / "boxes_per_match.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    produced.append(p)

    # ---- 2. Box width / height histograms (percentile bars) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI)
    for ax, dim, title in zip(axes, ["box_width_px", "box_height_px"], ["Box Width (px)", "Box Height (px)"]):
        p05_vals = [matches_data[m].get(dim, {}).get("p05", 0) for m in match_ids]
        med_vals = [matches_data[m].get(dim, {}).get("median", 0) for m in match_ids]
        p95_vals = [matches_data[m].get(dim, {}).get("p95", 0) for m in match_ids]
        x = range(len(match_ids))
        ax.bar(x, p95_vals, label="p95", color="#BBDEFB", edgecolor="#1565C0")
        ax.bar(x, med_vals, label="median", color="#42A5F5", edgecolor="#1565C0")
        ax.bar(x, p05_vals, label="p05", color="#1565C0", edgecolor="#0D47A1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(match_ids, rotation=30, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Pixels")
        ax.legend(fontsize=8)
        # Paper reference
        paper_ref = stats.get("paper_reference", {}).get("median_box_wh_px", [None, None])
        ref_val = paper_ref[0] if dim == "box_width_px" else paper_ref[1]
        if ref_val:
            ax.axhline(ref_val, color="red", linestyle="--", linewidth=1, label=f"Paper median ({ref_val}px)")
            ax.legend(fontsize=8)
    fig.suptitle("Box Size Distribution by Match", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "box_size_hist.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    produced.append(p)

    # ---- 3. Height vs image row (perspective effect) -----------------------
    if box_sample is not None and not box_sample.empty:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=DPI)
        # foot y position = y + h
        y_foot = box_sample["y"] + box_sample["h"]
        ax.scatter(y_foot, box_sample["h"], alpha=0.3, s=5, color="#1976D2")
        ax.set_xlabel("Foot y-coordinate (px from top)")
        ax.set_ylabel("Box height (px)")
        ax.set_title("Box Height vs. Image Row (Perspective Effect)", fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = out_dir / "height_vs_row.png"
        fig.savefig(p, dpi=DPI)
        plt.close(fig)
        produced.append(p)

    # ---- 4. Track length distribution (min/median/max per match) -----------
    fig, ax = plt.subplots(figsize=(max(6, len(match_ids) * 0.9), 4), dpi=DPI)
    tl_min = [matches_data[m].get("track_length_frames", {}).get("min", 0) for m in match_ids]
    tl_med = [matches_data[m].get("track_length_frames", {}).get("median", 0) for m in match_ids]
    tl_max = [matches_data[m].get("track_length_frames", {}).get("max", 0) for m in match_ids]
    x = range(len(match_ids))
    ax.bar(x, tl_max, label="max", color="#C8E6C9", edgecolor="#2E7D32")
    ax.bar(x, tl_med, label="median", color="#66BB6A", edgecolor="#2E7D32")
    ax.bar(x, tl_min, label="min", color="#2E7D32", edgecolor="#1B5E20")
    ax.set_xticks(list(x))
    ax.set_xticklabels(match_ids, rotation=30, fontsize=7)
    ax.set_title("Track Length Distribution per Match", fontsize=11, fontweight="bold")
    ax.set_xlabel("Match ID")
    ax.set_ylabel("Frames")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "track_length.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    produced.append(p)

    # ---- 5. Data quality issues table --------------------------------------
    issue_keys = ["degenerate_boxes", "clipped_boxes", "duplicate_rows", "frames_not_expected"]
    fig, ax = plt.subplots(figsize=(max(8, len(match_ids) * 1.1), 2.5), dpi=DPI)
    ax.axis("off")
    table_data = [
        [matches_data[m].get("issues", {}).get(k, 0) for k in issue_keys]
        for m in match_ids
    ]
    col_labels = ["Degen. Boxes", "Clipped", "Duplicate Rows", "Frames ≠ 22"]
    tbl = ax.table(
        cellText=table_data,
        rowLabels=match_ids,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.4)
    # Highlight non-zero cells
    for (row, col), cell in tbl.get_celld().items():
        if row > 0 and col >= 0:
            val = table_data[row - 1][col] if col < len(issue_keys) else 0
            if val and val != 0:
                cell.set_facecolor("#FFCDD2")
    ax.set_title("Data Quality Issues per Match", fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    p = out_dir / "data_issues.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    produced.append(p)

    return produced


# ---------------------------------------------------------------------------
# M2: side_by_side
# ---------------------------------------------------------------------------

def side_by_side(
    frame_bgr: "np.ndarray",
    left: "pd.DataFrame",
    right: "pd.DataFrame",
    frame_id: int,
    titles: tuple[str, str],
) -> "np.ndarray":
    """Stack two overlays (e.g. Pipeline A vs Pipeline B) **vertically**.

    For wide 4096 × 1080 panoramas, horizontal stacking would produce a
    4096 × 2160 image that is impractical to display.  Vertical stacking at
    the original resolution keeps each panel legible.

    Parameters
    ----------
    frame_bgr:
        Original BGR frame shared by both overlays.
    left, right:
        Track or Detection tables for the left (top) and right (bottom) panels.
    frame_id:
        0-based frame index.
    titles:
        ``(top_title, bottom_title)`` drawn at the top of each panel.
    """
    import cv2  # type: ignore
    import numpy as np

    top = draw_boxes(frame_bgr, left, frame_id)
    bottom = draw_boxes(frame_bgr, right, frame_id)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    pad = 4
    for panel, title in [(top, titles[0]), (bottom, titles[1])]:
        (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
        cv2.rectangle(panel, (0, 0), (tw + 2 * pad, th + 2 * pad), (30, 30, 30), -1)
        cv2.putText(panel, title, (pad, th + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.vstack([top, bottom])


# ---------------------------------------------------------------------------
# M2: overlay_video
# ---------------------------------------------------------------------------

def overlay_video(
    clip_path: Path,
    tables: dict[str, "pd.DataFrame"],
    frames: range,
    out_path: Path,
    scale: float = 0.5,
) -> Path:
    """Write an MP4 overlay (git-ignored) for the given frames.

    Frames are decoded sequentially with ``data_io.iter_frames``.  Each table
    in *tables* is overlaid in order; the dict key is used as the legend label.
    The output is scaled by *scale* (0.5 → half resolution) so that
    4096 × 1080 panoramas become 2048 × 540 for the presentation.

    Parameters
    ----------
    clip_path:
        Path to the MP4 video file.
    tables:
        Mapping of ``label → Track/Detection table`` to overlay.
    frames:
        Frame indices to include (from ``frameset_range``).
    out_path:
        Destination ``.mp4`` file (parent is created if needed).
    scale:
        Spatial scale factor (default 0.5 for panoramic clips).
    """
    import cv2  # type: ignore
    import sys

    # Import data_io from the src package
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        return out_path

    try:
        from soccertrack.data_io import iter_frames  # type: ignore
        frame_iter = iter_frames(clip_path, list(frames))
    except (ImportError, NotImplementedError):
        # Fallback: direct cv2 sequential read when Member 1's iter_frames is not yet ready
        frame_iter = _fallback_iter_frames(clip_path, frames)

    writer = None
    for frame_id, bgr in frame_iter:
        # Draw all table overlays
        canvas = bgr
        for label, table in tables.items():
            canvas = draw_boxes(canvas, table, frame_id)

        # Scale
        if scale != 1.0:
            new_w = int(canvas.shape[1] * scale)
            new_h = int(canvas.shape[0] * scale)
            canvas = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 25.0, (w, h))

        writer.write(canvas)

    if writer is not None:
        writer.release()

    return out_path


def _fallback_iter_frames(clip_path: Path, frames: range):
    """Fallback sequential frame decoder used when data_io.iter_frames is not available."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(clip_path))
    frame_set = set(frames)
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx in frame_set:
            yield idx, bgr
        idx += 1
        if idx > max(frames):
            break
    cap.release()


# ---------------------------------------------------------------------------
# M2: plot_stage_figure
# ---------------------------------------------------------------------------

def plot_stage_figure(stage_dir: Path, frame_id: int, out_path: Path) -> Path:
    """Compose the four stage images into one 2×2 panel report figure.

    Expected files in *stage_dir*::

        <frame_id:06d>_frame.png
        <frame_id:06d>_foreground.png
        <frame_id:06d>_morph.png
        <frame_id:06d>_proposals.png

    If any file is missing the panel is replaced by a placeholder.

    Parameters
    ----------
    stage_dir:
        ``outputs/stages/<match>/`` directory produced by Member 1.
    frame_id:
        0-based frame index.
    out_path:
        Destination PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stage_dir = Path(stage_dir)
    panels = [
        (f"{frame_id:06d}_frame.png",       "Original Frame"),
        (f"{frame_id:06d}_foreground.png",   "Foreground Mask"),
        (f"{frame_id:06d}_morph.png",        "After Morphology"),
        (f"{frame_id:06d}_proposals.png",    "Region Proposals"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6), dpi=150)
    for ax, (fname, title) in zip(axes.ravel(), panels):
        fpath = stage_dir / fname
        if fpath.exists():
            img = mpimg.imread(str(fpath))
            ax.imshow(img)
        else:
            # Placeholder
            ax.set_facecolor("#EEEEEE")
            ax.text(0.5, 0.5, f"[{fname}\nnot found]", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle(f"Segmentation Pipeline — Frame {frame_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Phase 3 stubs (implemented in Phase 3)
# ---------------------------------------------------------------------------

def plot_trajectories(pitch_tracks: "pd.DataFrame", out_path: Path) -> Path:
    """Phase 3: player trajectories on a 105 × 68 m pitch drawing.

    Draws each player's path on a schematic pitch (goal lines, centre circle,
    penalty areas).  One colour per ``track_id`` using the same stable hash
    scheme as :func:`draw_boxes`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    _draw_pitch(ax)

    for tid, grp in pitch_tracks.groupby("track_id"):
        color = [c / 255.0 for c in reversed(_stable_color_bgr(int(tid)))]  # BGR→RGB
        ax.plot(grp["x_m"], grp["y_m"], color=color, linewidth=1.0, alpha=0.8)
        # Mark start
        ax.scatter(grp["x_m"].iloc[0], grp["y_m"].iloc[0], color=color, s=20, zorder=5)

    ax.set_title("Player Trajectories (pitch coordinates)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_heatmap(
    pitch_tracks: "pd.DataFrame",
    track_id: int,
    out_path: Path,
    bins: tuple[int, int] = (21, 14),
) -> Path:
    """Phase 3: occupancy heatmap of one player on the pitch.

    Parameters
    ----------
    pitch_tracks:
        PitchTrack table (columns: frame, track_id, x_m, y_m, quality).
    track_id:
        Player ID to visualise.
    out_path:
        Destination PNG path.
    bins:
        ``(nx, ny)`` histogram bins across the 105 × 68 m pitch.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    player = pitch_tracks[pitch_tracks["track_id"] == track_id]

    fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
    _draw_pitch(ax)

    if not player.empty:
        h, xedges, yedges = np.histogram2d(
            player["x_m"], player["y_m"],
            bins=bins,
            range=[[-52.5, 52.5], [-34, 34]],
        )
        ax.imshow(
            h.T,
            origin="lower",
            extent=[-52.5, 52.5, -34, 34],
            aspect="auto",
            cmap="hot",
            alpha=0.65,
            interpolation="bilinear",
        )

    ax.set_title(f"Heatmap — Player {track_id}", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Internal pitch drawing helper
# ---------------------------------------------------------------------------

def _draw_pitch(ax) -> None:
    """Draw a schematic 105 × 68 m football pitch on *ax*."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    pitch_color = "#4CAF50"
    line_color = "white"
    lw = 1.5

    # Pitch outline
    ax.set_facecolor(pitch_color)
    ax.set_xlim(-57, 57)
    ax.set_ylim(-36, 36)
    ax.add_patch(mpatches.Rectangle((-52.5, -34), 105, 68, linewidth=lw,
                                     edgecolor=line_color, facecolor="none"))
    # Halfway line
    ax.axvline(0, color=line_color, linewidth=lw)
    # Centre circle
    ax.add_patch(mpatches.Circle((0, 0), 9.15, linewidth=lw,
                                  edgecolor=line_color, facecolor="none"))
    ax.scatter([0], [0], color=line_color, s=20, zorder=5)
    # Penalty areas
    for sign in (-1, 1):
        x0 = sign * 52.5
        x1 = x0 - sign * 16.5
        ax.add_patch(mpatches.Rectangle((min(x0, x1), -20.16),
                                         16.5, 40.32, linewidth=lw,
                                         edgecolor=line_color, facecolor="none"))
        ax.add_patch(mpatches.Rectangle((min(x0, x1) + (0 if sign == 1 else 10.97), -9.16),
                                         5.5, 18.32, linewidth=lw,
                                         edgecolor=line_color, facecolor="none"))
