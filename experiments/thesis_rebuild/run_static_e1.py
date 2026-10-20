"""E1: static validation of APETE + MMD-Critic against a random criticism control.

The underlying stationary sample is held fixed across model seeds so index-based
stability is meaningful. For each seed, APETE is fit to a seeded Random Forest;
MMD-Critic and random baselines then select the same K number of non-prototype
instances (unless fewer candidates remain).

Run from the Protree repo root:
  python -m experiments.thesis_rebuild.run_static_e1 --output results/thesis_rebuild_e1 --run-id e1run001
"""
from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

from experiments.thesis_rebuild.controlled_streams import Scenario, generate_controlled_stream


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _global_proto_indices(prototypes) -> list[int]:
    ids = []
    for points in prototypes.values():
        if points is None:
            continue
        if isinstance(points, pd.DataFrame):
            ids.extend(points.index.tolist())
        else:
            frame = pd.DataFrame(points)
            ids.extend(frame.index.tolist())
    return sorted(set(int(i) for i in ids))


def _nearest_neighbor_predict(X_query, X_anchors, y_anchors):
    d = pairwise_distances(np.asarray(X_query, dtype=float), np.asarray(X_anchors, dtype=float), metric="euclidean")
    return np.asarray(y_anchors)[np.argmin(d, axis=1)]


def _coverage(X, X_repr):
    d = pairwise_distances(np.asarray(X, dtype=float), np.asarray(X_repr, dtype=float), metric="euclidean").min(axis=1)
    return {
        "coverage_mean_distance": float(np.mean(d)),
        "coverage_median_distance": float(np.median(d)),
        "coverage_p90_distance": float(np.quantile(d, 0.90)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-trees", type=int, default=300)
    parser.add_argument("--k-criticisms", default="1,3,5,10")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1001)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--output", default="results/thesis_rebuild_e1")
    parser.add_argument("--run-id", default="e1run001")
    args = parser.parse_args()

    from protree.explainers.tree_distance import APete
    from protree.criticism.mmd_critic import MMDCritic

    if args.n_train < 100 or args.n_test < 100:
        parser.error("n-train and n-test must be at least 100")
    k_values = [int(v) for v in args.k_criticisms.split(",")]
    if any(k < 1 for k in k_values):
        parser.error("all K values must be >= 1")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    n_total = args.n_train + args.n_test
    stationary = Scenario(
        scope="global", drift_kind="stationary", schedule="abrupt", extent=1.0,
        magnitude="medium", drift_start=n_total + 1, transition_width=1,
    )
    static = generate_controlled_stream(n_total, args.data_seed, stationary)
    feature_cols = ["x1", "x2"]
    X_train_df = static.iloc[:args.n_train][feature_cols].copy().reset_index(drop=True)
    y_train = static.iloc[:args.n_train]["y"].to_numpy(dtype=int)
    X_test = static.iloc[args.n_train:][feature_cols].to_numpy(dtype=float)
    y_test = static.iloc[args.n_train:]["y"].to_numpy(dtype=int)
    static_test_meta = static.iloc[args.n_train:][["t", "x1", "x2", "y"]].copy()

    result_rows, member_rows = [], []
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    manifest = {
        "run_id": args.run_id,
        "experiment": "E1 static criticism validation",
        "n_train": args.n_train,
        "n_test": args.n_test,
        "n_trees": args.n_trees,
        "k_criticisms": k_values,
        "seeds": seeds,
        "data_seed_fixed_across_model_seeds": args.data_seed,
        "features": feature_cols,
        "evaluation": "1-nearest-exemplar proxy; fidelity = agreement with RF prediction on held-out samples",
        "software_versions": {"python": __import__("platform").python_version(), "river": _version("river"), "protree": _version("protree"), "numpy": _version("numpy"), "pandas": _version("pandas"), "scikit_learn": _version("scikit-learn")},
        "git_commit": _git_commit(),
        "warning": "The nearest-exemplar evaluation is an operational E1 proxy, not a claim of equivalence to every fidelity metric in the RACE-P paper.",
    }
    (outdir / f"manifest_{args.run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for repeat, seed in enumerate(seeds, start=1):
        np.random.seed(seed)
        model = RandomForestClassifier(n_estimators=args.n_trees, random_state=seed, n_jobs=-1)
        model.fit(X_train_df, y_train)
        explainer = APete(model=model, alpha=0.01)
        prototypes = explainer.select_prototypes(X_train_df, y_train)
        proto_global = _global_proto_indices(prototypes)
        proto_positions = np.asarray(X_train_df.index.get_indexer(proto_global), dtype=int)
        if not len(proto_positions) or np.any(proto_positions < 0):
            raise ValueError(f"Invalid APETE prototype indices for seed={seed}: {proto_global}")
        proto_set = set(proto_positions.tolist())
        eligible = np.asarray([i for i in range(len(X_train_df)) if i not in proto_set], dtype=int)
        if not len(eligible):
            raise ValueError("APETE selected every training sample; no criticism candidates remain.")

        X_train = X_train_df.to_numpy(dtype=float)
        X_proto = X_train[proto_positions]
        y_proto = y_train[proto_positions]
        rf_pred_test = model.predict(X_test)
        p_pred_test = _nearest_neighbor_predict(X_test, X_proto, y_proto)
        p_cov = _coverage(X_train, X_proto)
        base_fidelity = float(np.mean(p_pred_test == rf_pred_test))
        base_accuracy = float(np.mean(p_pred_test == y_test))

        for proto_pos in proto_positions:
            member_rows.append({
                "seed": seed, "repeat": repeat, "role": "prototype", "method": "APETE",
                "k_criticisms": 0, "train_position": int(proto_pos),
                "x1": float(X_train[proto_pos, 0]), "x2": float(X_train[proto_pos, 1]),
                "y": int(y_train[proto_pos]), "witness": np.nan, "score": np.nan,
            })

        # APETE-only reference row, emitted once per K so downstream comparisons line up.
        for k in k_values:
            result_rows.append({
                "seed": seed, "repeat": repeat, "method": "APETE_only", "k_criticisms": k,
                "n_prototypes": int(len(proto_positions)), "n_criticisms": 0,
                **p_cov,
                "coverage_with_criticism_mean_distance": p_cov["coverage_mean_distance"],
                "coverage_with_criticism_p90_distance": p_cov["coverage_p90_distance"],
                "heldout_model_agreement": base_fidelity,
                "heldout_label_accuracy": base_accuracy,
                "coverage_improvement_mean": 0.0,
                "fidelity_delta_vs_P": 0.0,
                "accuracy_delta_vs_P": 0.0,
            })

        for k in k_values:
            critic = MMDCritic(n_criticisms=k, gamma="scale")
            critic_positions, scores, witness = critic.select(
                X_train,
                prototype_indices=proto_positions.tolist(),
            )
            critic_positions = np.asarray(critic_positions, dtype=int)
            if len(set(critic_positions.tolist()) & proto_set):
                raise AssertionError("MMD-Critic returned a prototype as a criticism")

            rng = np.random.default_rng(seed * 1009 + k)
            random_positions = rng.choice(eligible, size=min(k, len(eligible)), replace=False)

            for method, positions, score_values, witness_values in (
                ("MMD_Critic", critic_positions, np.asarray(scores, dtype=float), np.asarray(witness, dtype=float)),
                ("Random", random_positions, np.full(len(random_positions), np.nan), np.full(len(random_positions), np.nan)),
            ):
                X_crit = X_train[positions]
                y_crit = y_train[positions]
                X_all = np.vstack([X_proto, X_crit])
                y_all = np.concatenate([y_proto, y_crit])
                joint_pred_test = _nearest_neighbor_predict(X_test, X_all, y_all)
                joint_cov = _coverage(X_train, X_all)
                c_dist = pairwise_distances(X_crit, X_proto).min(axis=1)
                row = {
                    "seed": seed, "repeat": repeat, "method": method, "k_criticisms": k,
                    "n_prototypes": int(len(proto_positions)), "n_criticisms": int(len(positions)),
                    **p_cov,
                    "coverage_with_criticism_mean_distance": joint_cov["coverage_mean_distance"],
                    "coverage_with_criticism_p90_distance": joint_cov["coverage_p90_distance"],
                    "heldout_model_agreement": float(np.mean(joint_pred_test == rf_pred_test)),
                    "heldout_label_accuracy": float(np.mean(joint_pred_test == y_test)),
                    "coverage_improvement_mean": p_cov["coverage_mean_distance"] - joint_cov["coverage_mean_distance"],
                    "fidelity_delta_vs_P": float(np.mean(joint_pred_test == rf_pred_test)) - base_fidelity,
                    "accuracy_delta_vs_P": float(np.mean(joint_pred_test == y_test)) - base_accuracy,
                    "criticism_mean_distance_to_P": float(np.mean(c_dist)) if len(c_dist) else np.nan,
                    "mean_abs_witness": float(np.mean(np.abs(witness_values))) if len(witness_values) and np.all(np.isfinite(witness_values)) else np.nan,
                }
                result_rows.append(row)
                for j, pos in enumerate(positions):
                    member_rows.append({
                        "seed": seed, "repeat": repeat, "role": "criticism", "method": method,
                        "k_criticisms": k, "train_position": int(pos),
                        "x1": float(X_train[pos, 0]), "x2": float(X_train[pos, 1]),
                        "y": int(y_train[pos]),
                        "witness": float(witness_values[j]) if j < len(witness_values) and np.isfinite(witness_values[j]) else np.nan,
                        "score": float(score_values[j]) if j < len(score_values) and np.isfinite(score_values[j]) else np.nan,
                    })

        print(f"[{repeat}/{len(seeds)}] seed={seed}: prototypes={len(proto_positions)}", flush=True)

        # Checkpoint so a failed later seed does not discard earlier results.
        pd.DataFrame(result_rows).to_csv(outdir / f"e1_static_metrics_{args.run_id}.csv", index=False)
        pd.DataFrame(member_rows).to_csv(outdir / f"e1_static_members_{args.run_id}.csv", index=False)

    print(f"Saved E1 outputs to {outdir}")


if __name__ == "__main__":
    main()
