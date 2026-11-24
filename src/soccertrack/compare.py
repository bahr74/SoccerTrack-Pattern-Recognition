"""Model comparison: Metrics JSON -> comparison tables (CSV) and evaluation figures.

Owner: Member 3. Inputs: outputs/metrics/*.json only. Outputs: outputs/tables/*.csv, outputs/figures/eval/*.png.
Member 4 embeds these outputs in the notebook, report and slides; Member 4 does not recompute metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# Which pipeline family a method belongs to (fair comparison: A = from scratch, B = pretrained YOLO).
def approach_of(method: str) -> str:
    detector = method.split("+")[0]
    if detector.startswith("scratch"):
        return "A (from scratch)"
    if detector.startswith("yolo"):
        return "B (YOLO)"
    return "oracle"


def collect_metrics(metrics_dir: Path) -> "pd.DataFrame":
    """Flatten all Metrics JSON files into one long table (one row per run_id x metric)."""
    import json

    import pandas as pd

    from soccertrack import contracts

    rows = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            doc = contracts.validate_metrics(json.load(f))
        base = {k: doc[k] for k in ("run_id", "eval_id", "task", "method", "split", "match", "frameset")}
        for metric, value in doc["metrics"].items():
            rows.append(dict(base, metric=metric, value=value))
    columns = ["run_id", "eval_id", "task", "method", "split", "match", "frameset", "metric", "value"]
    return pd.DataFrame(rows, columns=columns)


def _wide(long_df: "pd.DataFrame", split: str, eval_ids: tuple[str, ...], metrics: dict[str, str]) -> "pd.DataFrame":
    """Pivot the long table to one row per run, keep and rename the requested metrics, sort pooled rows last."""
    import pandas as pd

    df = long_df[(long_df["split"] == split) & (long_df["eval_id"].isin(eval_ids))]
    if df.empty:
        return pd.DataFrame(columns=["eval_id", "approach", "method", "match", "frameset", *metrics.values()])
    # One row per run that actually exists (pivot, not pivot_table, which would add every combination).
    wide = df.pivot(index=["eval_id", "method", "match", "frameset"], columns="metric", values="value").reset_index()
    wide.columns.name = None
    for key in metrics:
        if key not in wide:
            wide[key] = None
    wide.insert(1, "approach", wide["method"].map(approach_of))
    wide["_pooled"] = wide["match"] == "pooled"
    wide = wide.sort_values(["eval_id", "approach", "method", "_pooled", "match"]).drop(columns="_pooled")
    out = wide[["eval_id", "approach", "method", "match", "frameset", *metrics]].rename(columns=metrics)
    return out.reset_index(drop=True)


def detection_table(long_df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """Approach A vs B detection table (per match + pooled): P, R, F1, AP50, AP50-95, FP/frame, FPS."""
    return _wide(long_df, split, ("E3",), {"precision": "P", "recall": "R", "f1": "F1", "ap50": "AP50",
                                           "ap50_95": "AP50-95", "fp_per_frame": "FP/frame", "fps": "FPS"})


def classifier_table(long_df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """E1 (balanced patches) and E2 (real proposals) for each Approach A classifier."""
    return _wide(long_df, split, ("E1", "E2"), {"sensitivity": "Sensitivity", "specificity": "Specificity",
                                                "precision": "Precision", "f1": "F1", "accuracy": "Accuracy",
                                                "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})


def tracking_table(long_df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """Detector x tracker table incl. oracle rows: HOTA, DetA, AssA, IDF1, MOTA, IDSW."""
    table = _wide(long_df, split, ("E4",), {"hota": "HOTA", "deta": "DetA", "assa": "AssA", "loca": "LocA",
                                            "idf1": "IDF1", "mota": "MOTA", "idsw": "IDSW"})
    parts = table["method"].str.split("+", n=1, expand=True) if len(table) else None
    table.insert(3, "detector", parts[0] if parts is not None else [])
    table.insert(4, "tracker", parts[1] if parts is not None else [])
    table["IDSW"] = table["IDSW"].astype("Int64")  # a count, not a ratio
    return table


def reference_table() -> "pd.DataFrame":
    """Literature references with comparability labels (plan Section 3.5): never mixed with our numbers."""
    import pandas as pd

    rows = [
        ("SoccerTrack v2 paper: MOT subset statistics", "59,842 frames; 1,316,524 boxes; median box 25 x 55 px",
         "yes", "Reproducible from the MOT files; reported next to our own EDA statistics"),
        ("SoccerTrack Challenge 2025: top-5 HOTA (test M7 + M9)", "0.59 - 0.60", "approximately",
         "Same clips and test matches, but needs the full 4-minute clips; exact configuration not fully documented. "
         "Upper-bound reference for tracking only"),
        ("GTATrack paper: ByteTrack baseline HOTA (their validation)", "0.42", "indicative only",
         "Their own validation split, not the test set"),
        ("SoccerTrack v2 paper: GSR baseline GS-HOTA", "12.25 (all halves) / 10.96 (test halves)", "no",
         "Different task: pitch-space, attribute-aware game-state reconstruction over 45-minute halves"),
        ("SoccerTrack v2 paper: BAS mAP@1s", "0.222 / 0.599", "no", "Different task (ball action spotting)"),
    ]
    return pd.DataFrame(rows, columns=["reference", "reported value", "comparable with our numbers?", "why"])


def dataset_statistics_table(eda_stats: dict[str, Any]) -> "pd.DataFrame":
    """Our reproduced dataset statistics (Member 1's outputs/eda/stats.json) next to the paper's numbers."""
    import numpy as np
    import pandas as pd

    matches = eda_stats["matches"].values()
    paper = eda_stats.get("paper_reference", {})
    median_w = float(np.median([m["box_width_px"]["median"] for m in matches]))
    median_h = float(np.median([m["box_height_px"]["median"] for m in matches]))
    paper_wh = paper.get("median_box_wh_px", [None, None])
    rows = [("matches", eda_stats["totals"]["matches"], 10),
            ("annotated frames", eda_stats["totals"]["frames"], paper.get("frames")),
            ("boxes", eda_stats["totals"]["boxes"], paper.get("boxes")),
            ("median box width (px)", median_w, paper_wh[0]),
            ("median box height (px)", median_h, paper_wh[1])]
    out = pd.DataFrame(rows, columns=["statistic", "ours (matches we use)", "paper (all 10 MOT clips)"])
    out["note"] = ["paper counts all 10 clips; we use the matches of our split"] * 3 + \
                  ["median of per-match medians"] * 2
    return out


def _figure(out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    return plt


def plot_confusion_matrix(metrics: dict[str, Any], out_path: Path) -> Path:
    """Confusion-matrix figure for an E1/E2 Metrics JSON."""
    import numpy as np

    plt = _figure(out_path)
    c = metrics["counts"]
    matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Blues")
    labels = ["background", "player"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=12)
    ax.set_title(f"{metrics['eval_id']} {metrics['method']} ({metrics['split']}, threshold "
                 f"{metrics['params'].get('score_threshold')})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_curves(metrics_list: list[dict[str, Any]], curve: str, out_path: Path) -> Path:
    """Overlay PR or ROC curves of several runs (curve = "pr" | "roc")."""
    plt = _figure(out_path)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    for doc in metrics_list:
        data = (doc.get("curves") or {}).get(curve)
        if not data:
            continue
        label = f"{doc['method']} [{doc['match']}]"
        if curve == "pr":
            score = doc["metrics"].get("ap50", doc["metrics"].get("pr_auc"))
            ax.step(data["recall"], data["precision"], where="post",
                    label=f"{label} AP={score:.3f}" if score is not None else label)
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
        else:
            score = doc["metrics"].get("roc_auc")
            ax.plot(data["fpr"], data["tpr"], label=f"{label} AUC={score:.3f}" if score is not None else label)
            ax.plot([0, 1], [0, 1], color="0.7", linewidth=0.8, linestyle="--")
            ax.set_xlabel("false positive rate (1 - specificity)")
            ax.set_ylabel("true positive rate (sensitivity)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{curve.upper()} curves", fontsize=10)
    ax.legend(fontsize=7, loc="lower left" if curve == "pr" else "lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_recall_by_height(metrics_list: list[dict[str, Any]], out_path: Path) -> Path:
    """Grouped bars: E3 recall per GT box-height bin for each detector (small, far players are the hard case)."""
    import numpy as np

    plt = _figure(out_path)
    docs = [d for d in metrics_list if (d.get("breakdown") or {}).get("recall_by_height_px")]
    bins = list(docs[0]["breakdown"]["recall_by_height_px"]) if docs else []
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    width = 0.8 / max(len(docs), 1)
    for i, doc in enumerate(docs):
        values = [doc["breakdown"]["recall_by_height_px"].get(b) for b in bins]
        ax.bar(np.arange(len(bins)) + i * width, [v if v is not None else 0 for v in values], width,
               label=f"{doc['method']} [{doc['match']}]")
    ax.set_xticks(np.arange(len(bins)) + width * (len(docs) - 1) / 2, bins)
    ax.set_xlabel("GT box height (px)")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def write_comparison(metrics_dir: Path, split: str, tables_dir: Path, figures_dir: Path,
                     eda_stats_path: Path | None = None) -> list[Path]:
    """Build every comparison table (CSV) and evaluation figure (PNG) from the Metrics JSON files."""
    import json

    from soccertrack import contracts

    written = []
    long_df = collect_metrics(metrics_dir)
    tables = {"detection": detection_table(long_df, split), "classifiers": classifier_table(long_df, split),
              "tracking": tracking_table(long_df, split), "references": reference_table()}
    if eda_stats_path is not None and Path(eda_stats_path).exists():
        with open(eda_stats_path, encoding="utf-8") as f:
            tables["dataset_statistics"] = dataset_statistics_table(json.load(f))
    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = Path(tables_dir) / f"{name}_{split}.csv" if name != "references" else Path(tables_dir) / "references.csv"
        table.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
        written.append(path)
    docs = []
    for path in sorted(Path(metrics_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            doc = contracts.validate_metrics(json.load(f))
        if doc["split"] == split:
            docs.append(doc)
    for doc in docs:
        if doc["eval_id"] in ("E1", "E2"):
            written.append(plot_confusion_matrix(doc, Path(figures_dir) / f"confusion_{doc['run_id']}.png"))
    detection = [d for d in docs if d["eval_id"] == "E3"]
    classification = [d for d in docs if d["eval_id"] in ("E1", "E2")]
    if detection:
        written.append(plot_curves(detection, "pr", Path(figures_dir) / f"pr_detection_{split}.png"))
        written.append(plot_recall_by_height(detection, Path(figures_dir) / f"recall_by_height_{split}.png"))
    if classification:
        written.append(plot_curves(classification, "roc", Path(figures_dir) / f"roc_classifiers_{split}.png"))
        written.append(plot_curves(classification, "pr", Path(figures_dir) / f"pr_classifiers_{split}.png"))
    return written
