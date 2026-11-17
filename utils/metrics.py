from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
except Exception:  # pragma: no cover
    precision_recall_fscore_support = None
    average_precision_score = None
    roc_auc_score = None


def z_norm(x: np.ndarray, mean: float, std: float, eps: float = 1e-8) -> np.ndarray:
    return (x - mean) / (std + eps)


def fuse_scores(
    rec: np.ndarray,
    short: np.ndarray,
    long: np.ndarray,
    rec_stat: tuple[float, float],
    short_stat: tuple[float, float],
    long_stat: tuple[float, float],
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    zr = z_norm(rec, rec_stat[0], rec_stat[1])
    zs = z_norm(short, short_stat[0], short_stat[1])
    zl = z_norm(long, long_stat[0], long_stat[1])
    return alpha * zr + beta * zs + gamma * zl


def _binary_auroc_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate(([0.0], tp / pos, [1.0]))
    fpr = np.concatenate(([0.0], fp / neg, [1.0]))
    return float(np.trapz(tpr, fpr))


def _binary_aupr_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = int((y_true == 1).sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    # Add endpoints for a valid PR curve integration.
    precision = np.concatenate(([1.0], precision, [precision[-1] if len(precision) else 0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    return float(np.trapz(precision, recall))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_score = np.nan_to_num(np.asarray(y_score, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    if precision_recall_fscore_support is None:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        aupr = _binary_aupr_manual(y_true, y_score)
        auroc = _binary_auroc_manual(y_true, y_score)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if average_precision_score is not None:
            aupr = float(average_precision_score(y_true, y_score))
            if np.isnan(aupr):
                aupr = _binary_aupr_manual(y_true, y_score)
        else:
            aupr = _binary_aupr_manual(y_true, y_score)
        try:
            if roc_auc_score is not None:
                auroc = float(roc_auc_score(y_true, y_score))
                if np.isnan(auroc):
                    auroc = _binary_auroc_manual(y_true, y_score)
            else:
                auroc = _binary_auroc_manual(y_true, y_score)
        except ValueError:
            auroc = _binary_auroc_manual(y_true, y_score)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "aupr": float(aupr),
        "auroc": float(auroc),
    }
