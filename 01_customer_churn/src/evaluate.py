"""Evaluation utilities for the churn project (step 5 of the PDF).

Computes ROC-AUC, PR-AUC, precision, recall, F1 at a chosen operating threshold, plus a
calibration assessment, and produces the comparison plots saved to ``outputs/``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ModelMetrics:
    name: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    threshold: float
    brier: float  # calibration error proxy (lower = better calibrated)

    def to_dict(self) -> dict:
        return asdict(self)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold on the PR curve that maximises F1."""
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )
    # thr has len(f1)-1 entries; ignore the last synthetic point
    f1_valid = f1[:-1] if len(f1) > len(thr) else f1
    if len(f1_valid) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1_valid))])


def evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray,
                   threshold: float | None = None) -> tuple[ModelMetrics, float]:
    thr = threshold if threshold is not None else best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    from sklearn.metrics import brier_score_loss, accuracy_score
    metrics = ModelMetrics(
        name=name,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        threshold=float(thr),
        brier=float(brier_score_loss(y_true, y_prob)),
    )
    return metrics, thr


def plot_roc_curves(results: dict[str, np.ndarray], y_true: np.ndarray, path) -> None:
    """results: {model_name: y_prob}"""
    plt.figure(figsize=(6, 5))
    for name, prob in results.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Churn Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_pr_curves(results: dict[str, np.ndarray], y_true: np.ndarray, path) -> None:
    plt.figure(figsize=(6, 5))
    for name, prob in results.items():
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        plt.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Churn Models")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_calibration(y_true: np.ndarray, prob_map: dict[str, np.ndarray], path,
                     n_bins: int = 10) -> None:
    """Reliability diagram: fraction of positives vs. predicted probability."""
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, prob in prob_map.items():
        frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", lw=2, label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves — Churn Models")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_feature_importance(importances: pd.Series, path, top_n: int = 20) -> None:
    imp = importances.sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(imp)), imp.values[::-1])
    plt.yticks(range(len(imp)), imp.index[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {len(imp)} Feature Importances")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def metrics_dataframe(metrics_list: list[ModelMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([m.to_dict() for m in metrics_list])
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
