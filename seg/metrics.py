"""Evaluation on whole sets of predictions (numpy).

Per channel IoU, mean IoU over the channels that are trained, and the two
safety numbers for a robot:

* ``obstacle_recall``: share of true obstacle cells predicted as any obstacle
  class. ``false_floor_rate`` = 1 - recall: obstacles the robot would drive into.
* ``floor_precision``: of the cells predicted floor, how many really are.

Cells with weight 0 (``ignore`` in exclude mode) are left out.
"""

from __future__ import annotations

import numpy as np


def confusion(target: np.ndarray, weight: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """(N, h, w, C) one-hot, (N, h, w), (N, h, w, C) -> (C, C) counts, rows = truth."""
    c = probs.shape[-1]
    keep = weight.reshape(-1) > 0
    t = target.reshape(-1, c).argmax(-1)[keep]
    p = probs.reshape(-1, c).argmax(-1)[keep]
    return np.bincount(t * c + p, minlength=c * c).reshape(c, c)


def summarize(cm: np.ndarray, channel_names: list[str], trained: list[int]) -> dict:
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    present = [c for c in trained if not np.isnan(iou[c])]

    total = cm.sum()
    fg_true = cm[1:, :].sum()
    fg_hit = cm[1:, 1:].sum()
    floor_pred = cm[:, 0].sum()
    recall = float(fg_hit / fg_true) if fg_true else float("nan")
    return {
        "cells": int(total),
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "iou": {channel_names[c]: _num(iou[c]) for c in range(len(channel_names))},
        "mean_iou": _num(np.mean(iou[present])) if present else None,
        "obstacle_recall": _num(recall),
        "false_floor_rate": _num(1.0 - recall),
        "floor_precision": _num(cm[0, 0] / floor_pred) if floor_pred else None,
        "confusion": {"labels": channel_names, "rows_truth_cols_pred": cm.tolist()},
    }


def _num(v) -> float | None:
    v = float(v)
    return None if np.isnan(v) else round(v, 4)


def format_summary(name: str, s: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{100 * v:.1f}%"
    ious = ", ".join(f"{k} {pct(v)}" for k, v in s["iou"].items() if v is not None)
    return (f"{name}: mean IoU {pct(s['mean_iou'])} ({ious}); "
            f"obstacle recall {pct(s['obstacle_recall'])}, "
            f"false floor {pct(s['false_floor_rate'])}, floor precision {pct(s['floor_precision'])}")
