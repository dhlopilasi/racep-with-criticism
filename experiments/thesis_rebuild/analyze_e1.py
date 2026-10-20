"""Summarize E1 static comparison metrics and index stability."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def summarize_members(members: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Repeat index-stability is meaningful because all repeats use the same static samples.
    for (method, role, k), g in members.groupby(["method", "role", "k_criticisms"], dropna=False):
        sets = {
            int(seed): set(part["train_position"].astype(int).tolist())
            for seed, part in g.groupby("seed")
        }
        overlaps = [_jaccard(sets[a], sets[b]) for a, b in combinations(sorted(sets), 2)]
        rows.append({
            "method": method,
            "role": role,
            "k_criticisms": int(k) if pd.notna(k) else 0,
            "n_seeds": len(sets),
            "mean_pairwise_jaccard": float(np.mean(overlaps)) if overlaps else np.nan,
            "std_pairwise_jaccard": float(np.std(overlaps, ddof=1)) if len(overlaps) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def paired_method_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired MMD-Critic vs random/APETE-only tests for each K."""
    rows = []
    measures = [
        ("coverage_with_criticism_mean_distance", "lower_is_better"),
        ("coverage_with_criticism_p90_distance", "lower_is_better"),
        ("heldout_model_agreement", "higher_is_better"),
        ("heldout_label_accuracy", "higher_is_better"),
    ]
    for k, group in metrics.groupby("k_criticisms", dropna=False):
        wide = group.pivot_table(index="seed", columns="method", values=[m for m, _ in measures], aggfunc="first")
        for measure, direction in measures:
            if measure not in wide.columns.get_level_values(0):
                continue
            methods = set(wide[measure].columns)
            comparisons = []
            if {"MMD_Critic", "Random"}.issubset(methods):
                comparisons.append(("MMD_Critic", "Random"))
            if {"MMD_Critic", "APETE_only"}.issubset(methods):
                comparisons.append(("MMD_Critic", "APETE_only"))
            for a, b in comparisons:
                pair = wide[measure][[a, b]].dropna()
                delta = (pair[a] - pair[b]).to_numpy(dtype=float)
                statistic, p_value = np.nan, np.nan
                if len(delta) >= 5 and np.any(delta != 0):
                    try:
                        test = wilcoxon(delta, alternative="two-sided")
                        statistic, p_value = float(test.statistic), float(test.pvalue)
                    except ValueError:
                        pass
                rows.append({
                    "k_criticisms": int(k) if pd.notna(k) else 0,
                    "metric": measure,
                    "comparison": f"{a}_minus_{b}",
                    "direction_of_improvement": direction,
                    "n_paired_seeds": int(len(delta)),
                    "mean_difference": float(np.mean(delta)) if len(delta) else np.nan,
                    "median_difference": float(np.median(delta)) if len(delta) else np.nan,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_unadjusted": p_value,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        # Holm correction across this family of E1 comparisons.
        p = result["wilcoxon_p_unadjusted"].to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(p))
        adjusted = np.full(p.shape, np.nan)
        ordered = valid[np.argsort(p[valid])]
        running = 0.0
        for rank, idx in enumerate(ordered):
            running = max(running, (len(ordered) - rank) * p[idx])
            adjusted[idx] = min(1.0, running)
        result["wilcoxon_p_holm"] = adjusted
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/thesis_rebuild_e1")
    parser.add_argument("--run-id", default="e1run001")
    args = parser.parse_args()
    directory = Path(args.input)
    metrics = pd.read_csv(directory / f"e1_static_metrics_{args.run_id}.csv")
    members = pd.read_csv(directory / f"e1_static_members_{args.run_id}.csv")
    summary = metrics.groupby(["method", "k_criticisms"], dropna=False).agg({
        "n_prototypes": ["mean", "std"],
        "n_criticisms": ["mean", "std"],
        "coverage_mean_distance": ["mean", "std"],
        "coverage_p90_distance": ["mean", "std"],
        "coverage_with_criticism_mean_distance": ["mean", "std"],
        "heldout_model_agreement": ["mean", "std"],
        "heldout_label_accuracy": ["mean", "std"],
        "coverage_improvement_mean": ["mean", "std"],
        "fidelity_delta_vs_P": ["mean", "std"],
        "accuracy_delta_vs_P": ["mean", "std"],
    }).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x != "") if isinstance(col, tuple) else str(col) for col in summary.columns]
    stability = summarize_members(members)
    tests = paired_method_tests(metrics)
    summary.to_csv(directory / f"e1_static_summary_{args.run_id}.csv", index=False)
    stability.to_csv(directory / f"e1_static_stability_{args.run_id}.csv", index=False)
    tests.to_csv(directory / f"e1_static_paired_tests_{args.run_id}.csv", index=False)
    print(summary.to_string(index=False))
    print("\nStability:\n", stability.to_string(index=False))
    print("\nPaired tests:\n", tests.to_string(index=False))


if __name__ == "__main__":
    main()
