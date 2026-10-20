"""Analyze E2 signal sensitivity, E4 detection/4-zone metrics and E7 localization.

Run from repo root after the controlled E2/E4 runner finishes:
  python -m experiments.thesis_rebuild.analyze_e2_e7 \
    --input results/thesis_rebuild_e2_e4 --run-id smoke001

Chunk-level Mann-Whitney p-values are exploratory because temporal chunks are
not independent. For final inference, emphasize independent-seed summaries and
paired tests across seeds; the script reports Holm-adjusted E7 p-values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from sklearn.metrics import roc_auc_score

SIGNALS = [
    "p_pri",
    "c_mean_abs_witness",
    "c_mean_nearest_prototype_distance",
    "c_criticism_reassignment_impact",
    "c_criticism_fraction",
    "c_random_mean_abs_witness",
    "c_error_mean_abs_witness",
]


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return np.nan
    return float((np.mean(b) - np.mean(a)) / pooled)


def _safe_auc(y_true, score):
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    y_true, score = y_true[valid], score[valid]
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true.astype(int), score))


def analyze_e2(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-stream AUC/effect sizes; retain seed rows for later aggregation."""
    rows = []
    group_cols = [
        c for c in [
            "scenario_id", "seed", "stream_scope", "drift_kind", "schedule",
            "extent", "magnitude", "chunk_size",
        ] if c in raw.columns
    ]
    for group_key, g in raw.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))
        y = g["gt_concept_drift"].astype(bool).to_numpy()
        has_concept_event = bool(y.any()) and bool((~y).any())
        for signal in SIGNALS:
            if signal not in g:
                continue
            scores = pd.to_numeric(g[signal], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(scores)
            before = scores[valid & ~y]
            after = scores[valid & y]
            p_value = np.nan
            if len(before) and len(after):
                try:
                    p_value = float(mannwhitneyu(before, after, alternative="two-sided").pvalue)
                except ValueError:
                    pass

                        # Additional metrics around the drift onset.
            # These complement, rather than replace, full pre/post AUC.
            onset_metrics = {
                "first_drift_chunk": np.nan,
                "onset_signal_value": np.nan,
                "onset_minus_pre_mean": np.nan,
                "onset_z_from_pre": np.nan,
                "first3_drift_chunks_mean": np.nan,
                "first3_minus_pre_mean": np.nan,
            }

            if has_concept_event and "chunk" in g.columns and len(before):
                drift_positions = np.flatnonzero(y)

                if len(drift_positions):
                    onset_pos = drift_positions[0]
                    onset_chunk = g.iloc[onset_pos]["chunk"]
                    onset_score = scores[onset_pos]
                    pre_mean = float(np.mean(before))

                    onset_metrics["first_drift_chunk"] = int(onset_chunk)

                    if np.isfinite(onset_score):
                        onset_metrics["onset_signal_value"] = float(onset_score)
                        onset_metrics["onset_minus_pre_mean"] = (
                            float(onset_score - pre_mean)
                        )

                    # Standardize onset relative to pre-drift variability.
                    if len(before) >= 2:
                        pre_std = float(np.std(before, ddof=1))

                        if pre_std > 0 and np.isfinite(onset_score):
                            onset_metrics["onset_z_from_pre"] = (
                                float((onset_score - pre_mean) / pre_std)
                            )

                    # Mean over the first three chunks labelled as
                    # concept-drift-active, excluding missing signal values.
                    early_scores = scores[drift_positions[:3]]
                    early_scores = early_scores[np.isfinite(early_scores)]

                    if len(early_scores):
                        early_mean = float(np.mean(early_scores))
                        onset_metrics["first3_drift_chunks_mean"] = early_mean
                        onset_metrics["first3_minus_pre_mean"] = (
                            float(early_mean - pre_mean)
                        )

            rows.append({
                **meta,
                **onset_metrics,
                "signal": signal,
                "n_before": int(len(before)),
                "n_during_after": int(len(after)),
                "pre_mean": float(np.mean(before)) if len(before) else np.nan,
                "post_mean": float(np.mean(after)) if len(after) else np.nan,
                "post_minus_pre": float(np.mean(after) - np.mean(before)) if len(before) and len(after) else np.nan,
                "cohens_d_post_vs_pre": _cohens_d(before, after),
                "roc_auc_concept_drift": _safe_auc(y, scores) if has_concept_event else np.nan,
                "mann_whitney_p_unadjusted": p_value,
                "auc_not_defined_no_concept_change": not has_concept_event,
            })
    return pd.DataFrame(rows)


def aggregate_e2(e2: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent stream/seed rows by controlled configuration."""
    if e2.empty:
        return pd.DataFrame()
    group_cols = [
        c for c in [
            "stream_scope", "drift_kind", "schedule", "extent", "magnitude",
            "chunk_size", "signal",
        ] if c in e2.columns
    ]
    metric_cols = [
        c for c in [
            "roc_auc_concept_drift",
            "cohens_d_post_vs_pre",
            "post_minus_pre",
            "onset_signal_value",
            "onset_minus_pre_mean",
            "onset_z_from_pre",
            "first3_drift_chunks_mean",
            "first3_minus_pre_mean",
        ]
        if c in e2.columns
    ]
    out = e2.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in col if part != "") if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def analyze_e7(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-seed localization rates after onset for local concept drift."""
    required = {
        "gt_concept_drift", "stream_scope", "drift_kind",
        "criticisms_in_affected_region_fraction",
        "current_prototypes_in_affected_region_fraction",
        "c_random_in_affected_region_fraction",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw signals missing E7 columns: {sorted(missing)}")
    local = raw[
        raw["gt_concept_drift"].astype(bool)
        & raw["stream_scope"].eq("local")
        & raw["drift_kind"].eq("concept")
    ].copy()
    keys = [c for c in ["scenario_id", "seed", "extent", "magnitude", "schedule", "chunk_size"] if c in local.columns]
    if local.empty:
        return pd.DataFrame(columns=keys + [
            "criticism_region_fraction", "prototype_region_fraction", "random_region_fraction",
            "criticism_minus_prototype", "criticism_minus_random",
        ])
    aggregations = {
        "criticism_region_fraction": ("criticisms_in_affected_region_fraction", "mean"),
        "prototype_region_fraction": ("current_prototypes_in_affected_region_fraction", "mean"),
        "random_region_fraction": ("c_random_in_affected_region_fraction", "mean"),
        "n_evaluated_chunks": ("chunk", "count"),
    }
    if "c_error_in_affected_region_fraction" in local.columns:
        aggregations["error_region_fraction"] = ("c_error_in_affected_region_fraction", "mean")
    grouped = local.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    grouped["criticism_minus_prototype"] = grouped["criticism_region_fraction"] - grouped["prototype_region_fraction"]
    grouped["criticism_minus_random"] = grouped["criticism_region_fraction"] - grouped["random_region_fraction"]
    if "error_region_fraction" in grouped.columns:
        grouped["criticism_minus_error"] = grouped["criticism_region_fraction"] - grouped["error_region_fraction"]
    return grouped


def _holm_adjust(p_values):
    """Holm family-wise correction that ignores NaN p-values."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if not len(valid_idx):
        return adjusted
    order = valid_idx[np.argsort(values[valid_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_e7_tests(e7: pd.DataFrame) -> pd.DataFrame:
    """Paired seed-level Wilcoxon tests by local-drift configuration."""
    if e7.empty:
        return pd.DataFrame()
    config_cols = [c for c in ["extent", "magnitude", "schedule", "chunk_size"] if c in e7.columns]
    comparisons = [
        ("criticism_minus_prototype", "criticism_vs_prototype"),
        ("criticism_minus_random", "criticism_vs_random"),
    ]
    if "criticism_minus_error" in e7.columns:
        comparisons.append(("criticism_minus_error", "criticism_vs_error_criticism"))
    rows = []
    for key, group in e7.groupby(config_cols, dropna=False):
        meta = dict(zip(config_cols, key if isinstance(key, tuple) else (key,)))
        for diff_col, comparison in comparisons:
            diffs = pd.to_numeric(group[diff_col], errors="coerce").dropna().to_numpy(dtype=float)
            statistic, p_value = np.nan, np.nan
            if len(diffs) >= 5 and np.any(diffs != 0):
                try:
                    result = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                except ValueError:
                    pass
            rows.append({
                **meta,
                "comparison": comparison,
                "n_seeds": int(len(diffs)),
                "mean_paired_difference": float(np.mean(diffs)) if len(diffs) else np.nan,
                "median_paired_difference": float(np.median(diffs)) if len(diffs) else np.nan,
                "std_paired_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else np.nan,
                "fraction_positive": float(np.mean(diffs > 0)) if len(diffs) else np.nan,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_unadjusted": p_value,
            })
    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = _holm_adjust(result["wilcoxon_p_unadjusted"].to_numpy())
    return result


def aggregate_detection(det: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        c for c in ["rule", "c_signal", "k", "w", "scope", "drift_kind", "extent",
                    "magnitude", "schedule", "chunk_size"] if c in det.columns
    ]
    metrics = [
        c for c in ["tp", "fn", "fp", "n_alarms", "n_true_events", "precision", "recall", "f1",
                    "mean_delay_chunks", "fp_per_1000_chunks", "four_zone_chunk_accuracy",
                    "four_zone_balanced_accuracy_supported_classes"]
        if c in det.columns
    ]
    agg = det.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else str(col) for col in agg.columns]
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/thesis_rebuild_e2_e4")
    parser.add_argument("--run-id", default="run001")
    args = parser.parse_args()
    input_dir = Path(args.input)
    raw_path = input_dir / f"raw_signals_{args.run_id}.csv"
    det_path = input_dir / f"detection_metrics_{args.run_id}.csv"
    zone_path = input_dir / f"four_zone_chunks_{args.run_id}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    raw = pd.read_csv(raw_path)
    e2 = analyze_e2(raw)
    e7 = analyze_e7(raw)
    e2_path = input_dir / f"e2_signal_sensitivity_{args.run_id}.csv"
    e2_agg_path = input_dir / f"e2_signal_sensitivity_aggregate_{args.run_id}.csv"
    e7_path = input_dir / f"e7_localization_{args.run_id}.csv"
    e7_test_path = input_dir / f"e7_paired_tests_{args.run_id}.csv"
    e2.to_csv(e2_path, index=False)
    aggregate_e2(e2).to_csv(e2_agg_path, index=False)
    e7.to_csv(e7_path, index=False)
    paired_e7_tests(e7).to_csv(e7_test_path, index=False)
    print(f"Saved: {e2_path}")
    print(f"Saved: {e2_agg_path}")
    print(f"Saved: {e7_path}")
    print(f"Saved: {e7_test_path}")

    if det_path.exists():
        det = pd.read_csv(det_path)
        det_agg_path = input_dir / f"detection_aggregate_{args.run_id}.csv"
        aggregate_detection(det).to_csv(det_agg_path, index=False)
        print(f"Saved: {det_agg_path}")

    if zone_path.exists():
        zones = pd.read_csv(zone_path)
        # Attach controlled condition metadata from the per-chunk raw signals.
        metadata_cols = [
            c for c in ["scenario_id", "seed", "chunk", "stream_scope", "drift_kind",
                        "schedule", "extent", "magnitude", "chunk_size"] if c in raw.columns
        ]
        join_keys = [c for c in ["scenario_id", "seed", "chunk"] if c in metadata_cols and c in zones.columns]
        meta = raw[metadata_cols].drop_duplicates(join_keys)
        zones = zones.merge(meta, on=join_keys, how="left", suffixes=("", "_raw"))
        case_keys = [
            c for c in ["stream_scope", "drift_kind", "schedule", "extent", "magnitude",
                        "chunk_size", "k", "c_signal", "ground_truth_class", "zone_prediction"]
            if c in zones.columns
        ]
        confusion = zones.groupby(case_keys, dropna=False).size().rename("n_chunks").reset_index()
        confusion_path = input_dir / f"four_zone_confusion_aggregate_{args.run_id}.csv"
        confusion.to_csv(confusion_path, index=False)
        # Keep the individual-stream version for diagnostics.
        individual_keys = [c for c in ["scenario_id", "seed", "k", "c_signal", "ground_truth_class", "zone_prediction"] if c in zones.columns]
        individual = zones.groupby(individual_keys, dropna=False).size().rename("n_chunks").reset_index()
        individual.to_csv(input_dir / f"four_zone_confusion_by_stream_{args.run_id}.csv", index=False)
        print(f"Saved: {confusion_path}")
        print(f"Saved: {input_dir / f'four_zone_confusion_by_stream_{args.run_id}.csv'}")


if __name__ == "__main__":
    main()
