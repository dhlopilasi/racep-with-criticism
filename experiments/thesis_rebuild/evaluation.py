"""Shared calibration, rule-combination, event and 4-zone evaluation helpers."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

ZONE_ORDER = ("none", "global", "local", "drift")


def fit_upper_threshold(values, reference_mask, k: float) -> dict:
    """Fit a separate upper threshold mu + k*sigma on reference-only chunks."""
    values = np.asarray(values, dtype=float)
    reference_mask = np.asarray(reference_mask, dtype=bool)
    valid = reference_mask & np.isfinite(values)
    ref = values[valid]
    if len(ref) < 2:
        return {"mean": np.nan, "std": np.nan, "threshold": np.nan, "n_reference": int(len(ref))}
    mu = float(np.mean(ref))
    sigma = float(np.std(ref, ddof=0))
    return {"mean": mu, "std": sigma, "threshold": mu + float(k) * sigma, "n_reference": int(len(ref))}


def above_threshold(values, threshold: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(threshold):
        return np.zeros(len(values), dtype=bool)
    return np.isfinite(values) & (values > threshold)


def combine_p_c(p_strong, c_strong, rule: str, w: int = 0) -> np.ndarray:
    """Combine signals online. AND alarms when the second signal arrives within w chunks.

    The AND rule is causal: at time t it only examines [t-w, t], not future chunks.
    """
    p = np.asarray(p_strong, dtype=bool)
    c = np.asarray(c_strong, dtype=bool)
    if p.shape != c.shape:
        raise ValueError("p_strong and c_strong must have the same shape")
    if w < 0:
        raise ValueError("w must be >= 0")
    if rule == "P":
        return p.copy()
    if rule == "C":
        return c.copy()
    if rule == "OR":
        return p | c
    if rule != "AND":
        raise ValueError("rule must be one of P, C, OR, AND")
    out = np.zeros(len(p), dtype=bool)
    for t in range(len(p)):
        start = max(0, t - w)
        seen_p = bool(np.any(p[start : t + 1]))
        seen_c = bool(np.any(c[start : t + 1]))
        # Alarm only on the chunk where the second signal arrives.
        out[t] = (p[t] and seen_c) or (c[t] and seen_p)
    return out


def four_zone(p_strong, c_strong) -> np.ndarray:
    """Operational four-zone mapping from the user's research plan.

    none: neither signal strong; global: P-only; local: C-only; drift: both.
    """
    p = np.asarray(p_strong, dtype=bool)
    c = np.asarray(c_strong, dtype=bool)
    if p.shape != c.shape:
        raise ValueError("p_strong and c_strong must have the same shape")
    labels = np.full(len(p), "none", dtype=object)
    labels[p & ~c] = "global"
    labels[~p & c] = "local"
    labels[p & c] = "drift"
    return labels


def event_detection_metrics(
    flags,
    true_event_starts: Iterable[int],
    n_chunks: int,
    tolerance: int,
) -> dict:
    """One-to-one event matching: detection must be in [start, start+tolerance]."""
    flags = np.asarray(flags, dtype=bool)
    if len(flags) != n_chunks:
        raise ValueError("flags length must equal n_chunks")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    alarms = list(np.flatnonzero(flags))
    starts = sorted(int(x) for x in true_event_starts)
    used_alarms: set[int] = set()
    delays: list[int] = []
    matched_events = 0

    for start in starts:
        valid = [a for a in alarms if a not in used_alarms and start <= a <= start + tolerance]
        if valid:
            first = min(valid)
            used_alarms.add(first)
            matched_events += 1
            delays.append(first - start)

    fp = len([a for a in alarms if a not in used_alarms])
    fn = len(starts) - matched_events
    precision = matched_events / (matched_events + fp) if matched_events + fp else 0.0
    recall = matched_events / len(starts) if starts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(matched_events),
        "fn": int(fn),
        "fp": int(fp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_delay_chunks": float(np.mean(delays)) if delays else np.nan,
        "fp_per_1000_chunks": float(fp * 1000.0 / max(n_chunks, 1)),
        "n_alarms": int(len(alarms)),
        "n_true_events": int(len(starts)),
    }


def four_zone_metrics(true_labels, predicted_labels) -> dict:
    true_labels = np.asarray(true_labels, dtype=object)
    predicted_labels = np.asarray(predicted_labels, dtype=object)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("true_labels and predicted_labels must have same shape")
    matrix = {gt: {pred: 0 for pred in ZONE_ORDER} for gt in ZONE_ORDER}
    for gt, pred in zip(true_labels, predicted_labels):
        g = gt if gt in ZONE_ORDER else "none"
        p = pred if pred in ZONE_ORDER else "none"
        matrix[g][p] += 1
    correct = sum(matrix[c][c] for c in ZONE_ORDER)
    n = int(len(true_labels))
    per_class_recall = {}
    for cls in ZONE_ORDER:
        support = sum(matrix[cls].values())
        per_class_recall[cls] = float(matrix[cls][cls] / support) if support else np.nan
    supported_recalls = [v for v in per_class_recall.values() if np.isfinite(v)]
    return {
        "accuracy": float(correct / n) if n else np.nan,
        "balanced_accuracy_supported_classes": float(np.mean(supported_recalls)) if supported_recalls else np.nan,
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix,
        "n_chunks": n,
    }


def majority_label(values: Iterable[str], default: str = "none") -> str:
    vals = [str(v) for v in values if v is not None]
    if not vals:
        return default
    counts = Counter(vals)
    top = max(counts.values())
    # Stable tie-break follows the planned semantic ordering.
    tied = {k for k, v in counts.items() if v == top}
    for label in ZONE_ORDER:
        if label in tied:
            return label
    return sorted(tied)[0]
