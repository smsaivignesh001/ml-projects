"""Evaluation for Predictive Maintenance (step 5 of the PDF).

Because failures are rare, the headline metrics are **recall** (did we catch failures?),
**precision / false alarms** (how many needless maintenance dispatches?), and **PR-AUC**
(which is far more informative than ROC-AUC under heavy imbalance). We also report the
absolute number of false alarms at the chosen operating point.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve, brier_score_loss,
)


@dataclass
class MaintMetrics:
    name: str
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    f1: float
    threshold: float
    true_positives: int
    false_positives: int     # = false alarms
    false_negatives: int     # = missed failures
    false_alarm_rate: float  # FP / (FP + TN)
    brier: float

    def to_dict(self):
        return asdict(self)


def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    f1_valid = f1[:-1] if len(f1) > len(thr) else f1
    if len(f1_valid) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1_valid))])


def evaluate_model(name, y_true, y_prob, threshold=None) -> tuple[MaintMetrics, float]:
    thr = threshold if threshold is not None else best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = (fp + tn)
    metrics = MaintMetrics(
        name=name,
        pr_auc=float(average_precision_score(y_true, y_prob)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(thr),
        true_positives=int(tp), false_positives=int(fp), false_negatives=int(fn),
        false_alarm_rate=float(fp / denom) if denom else 0.0,
        brier=float(brier_score_loss(y_true, y_prob)),
    )
    return metrics, thr


def metrics_dataframe(metrics_list) -> pd.DataFrame:
    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    return df.sort_values("pr_auc", ascending=False).reset_index(drop=True)


# ---------------- plots ----------------
def plot_pr_curves(results: dict, y_true, path):
    plt.figure(figsize=(6, 5))
    for name, prob in results.items():
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        plt.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    base = y_true.mean()
    plt.axhline(base, ls="--", color="grey", lw=1, label=f"baseline ({base:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall — Failure Prediction")
    plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_roc_curves(results: dict, y_true, path):
    plt.figure(figsize=(6, 5))
    for name, prob in results.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_true, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — Failure Prediction")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_false_alarm_tradeoff(y_true, y_prob, path, name="model"):
    """Precision, recall and false-alarm count as the decision threshold varies."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    thr = np.clip(thr, 0, 1)
    n_pos = int(y_true.sum())
    fps = []
    for t in thr:
        pred = (y_prob >= t).astype(int)
        fps.append(int(((pred == 1) & (y_true == 0)).sum()))
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(thr, prec[:-1], label="Precision", color="tab:blue", lw=2)
    ax1.plot(thr, rec[:-1], label="Recall", color="tab:green", lw=2)
    ax1.set_xlabel("Decision threshold"); ax1.set_ylabel("Score")
    ax2 = ax1.twinx()
    ax2.plot(thr, fps, label="False alarms", color="tab:red", ls="--", lw=2)
    ax2.set_ylabel("False alarms (count)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.title(f"Precision / Recall / False-Alarm Trade-off — {name}")
    fig.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Oranges")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["Pred OK", "Pred FAIL"]); ax.set_yticks([0, 1], ["True OK", "True FAIL"])
    ax.set_title(title); fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_feature_importance(importances: pd.Series, path, top_n=20):
    imp = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(imp)), imp.values[::-1], color="steelblue")
    plt.yticks(range(len(imp)), imp.index[::-1], fontsize=8)
    plt.xlabel("Importance"); plt.title(f"Top {len(imp)} Features")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
