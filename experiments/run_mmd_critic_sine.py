from asyncio import selector_events
from pathlib import Path

import numpy as np
import pandas as pd

from river.datasets import synth
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from protree.explainers.tree_distance import APete
from protree.criticism.mmd_critic import MMDCritic


# ============================================================
# Configuration
# ============================================================

TOTAL_N = 30_000
DRIFT_POSITION = 28_000

# Number of samples around the known drift
PRE_SIZE = 2_000
POST_SIZE = 2_000

N_CRITICISMS = 5

# Practical RBF setting.
# For strict reproduction of a particular paper experiment,
# replace this with the paper's exact kernel parameter.
GAMMA = "scale"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

N_REPEATS = 10
SEED_BASE = 1000

# ============================================================
# 1. Generate Sine stream
# ============================================================

def generate_sine(seed: int) -> pd.DataFrame:
    """
    Generate a Sine stream with one abrupt concept drift.

    The drift is introduced explicitly at DRIFT_POSITION by
    changing the Sine classification function.
    """

    ds = synth.Sine(
        classification_function=0,
        seed=seed,
        balance_classes=False,
        has_noise=False,
    )

    rows = []

    stream = iter(ds)

    for i in range(TOTAL_N):
        # Introduce the concept drift exactly at this position.
        if i == DRIFT_POSITION:
            ds.generate_drift()

        x, y = next(stream)

        row = dict(x)
        row["y"] = y
        row["t"] = i

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# 2. Split data around drift
# ============================================================

def split_pre_post(df: pd.DataFrame):
    """
    Take two windows around the known drift position.
    """

    pre_start = DRIFT_POSITION - PRE_SIZE
    pre_end = DRIFT_POSITION

    post_start = DRIFT_POSITION
    post_end = DRIFT_POSITION + POST_SIZE

    pre = df[
        (df["t"] >= pre_start)
        & (df["t"] < pre_end)
    ].copy()

    post = df[
        (df["t"] >= post_start)
        & (df["t"] < post_end)
    ].copy()

    return pre, post


# ============================================================
# 3. Train classifier
# ============================================================

def train_model(df: pd.DataFrame, seed: int):
    """
    Train a Random Forest on the selected window.
    """

    feature_cols = [
        c for c in df.columns
        if c not in {"y", "t"}
    ]

    X = df[feature_cols]
    y = df["y"]

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X, y)

    return model, feature_cols


# ============================================================
# 4. Select APete prototypes
# ============================================================

def select_prototypes(
    model,
    df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Select prototypes using APete.
    """

    explainer = APete(
        model=model,
        alpha=0.01,
    )

    X = df[feature_cols]
    y = df["y"].to_numpy()

    prototypes = explainer.select_prototypes(
        X,
        y,
    )

    return prototypes


# ============================================================
# 5. Prototype index handling
# ============================================================

def prototype_global_indices(prototypes) -> list[int]:
    """
    Return original DataFrame indices of APete prototypes.

    APete -> class filtering -> iloc() preserves the original
    DataFrame index, as verified in the previous test.
    """

    indices = []

    for _, proto_df in prototypes.items():
        if proto_df is None or len(proto_df) == 0:
            continue

        indices.extend(proto_df.index.tolist())

    return sorted(set(indices))


def prototype_position_indices(
    df: pd.DataFrame,
    prototypes,
) -> list[int]:
    """
    Convert original DataFrame indices to positional indices
    used by numpy arrays passed to MMDCritic.
    """

    global_indices = prototype_global_indices(prototypes)

    positions = df.index.get_indexer(global_indices)

    if np.any(positions < 0):
        missing = [
            idx
            for idx, pos in zip(global_indices, positions)
            if pos < 0
        ]
        raise ValueError(
            f"Prototype indices not found in dataframe: {missing}"
        )

    return positions.tolist()


# ============================================================
# 6. Run original MMD-Critic
# ============================================================

def run_mmd_critic(
    df: pd.DataFrame,
    prototypes,
    n_criticisms: int = N_CRITICISMS,
):
    """
    Run standalone MMD-Critic on the original feature space.

    Important:
    - df.index = global/original indices
    - X positions = 0..n-1
    - MMDCritic receives positional prototype indices
    """

    feature_cols = [
        c for c in df.columns
        if c not in {"y", "t"}
    ]

    X = df[feature_cols].to_numpy()

    proto_positions = prototype_position_indices(
        df,
        prototypes,
    )

    critic = MMDCritic(
        n_criticisms=n_criticisms,
        gamma=GAMMA,
    )

    criticisms, scores, witness = critic.select(
        X,
        prototype_indices=proto_positions,
    )

    return criticisms, scores, witness


# ============================================================
# 7. Convert MMD-Critic output positions -> global indices
# ============================================================

def criticism_global_indices(
    df: pd.DataFrame,
    criticism_positions,
) -> list:
    """
    MMDCritic works on numpy positions.
    Convert them back to original DataFrame indices.
    """

    return [
        df.index[pos]
        for pos in criticism_positions
    ]


# ============================================================
# 8. Save readable results
# ============================================================
def save_results(
    name: str,
    df: pd.DataFrame,
    prototypes,
    criticism_positions,
    scores,
    witness,
):
    rows = []

    # --------------------------------------------------------
    # Prototypes
    # --------------------------------------------------------

    proto_global = prototype_global_indices(
        prototypes
    )

    for idx in proto_global:
        row = df.loc[idx].to_dict()

        row["role"] = "prototype"
        row["mmd_score"] = np.nan
        row["witness"] = np.nan

        rows.append(row)

    # --------------------------------------------------------
    # Criticisms
    # --------------------------------------------------------

    criticism_global = criticism_global_indices(
        df,
        criticism_positions,
    )

    for position, global_idx, score, w in zip(
        criticism_positions,
        criticism_global,
        scores,
        witness,
    ):
        row = df.loc[global_idx].to_dict()

        row["role"] = "criticism"
        row["mmd_score"] = float(score)
        row["witness"] = float(w)

        rows.append(row)

    out = pd.DataFrame(rows)

    path = RESULTS_DIR / f"mmd_critic_sine_{name}.csv"

    out.to_csv(
        path,
        index=False,
    )

    print(f"Saved: {path}")

    return out

# ============================================================
# 9. Run one window
# ============================================================

def run_window(
    name: str,
    df: pd.DataFrame,
    seed: int,
    output_suffix: str,
): 
    """
    Train model -> APete prototypes -> MMD-Critic criticisms.
    """

    print(f"\n=== {name.upper()} ===")

    model, feature_cols = train_model(
        df,
        seed,
    )

    prototypes = select_prototypes(
        model,
        df,
        feature_cols,
    )

    proto_global = prototype_global_indices(
        prototypes
    )

    print(
        f"Number of prototypes: {len(proto_global)}"
    )

    criticisms, scores, witness = run_mmd_critic(
        df,
        prototypes,
    )

    print(
        "Criticism positions:",
        criticisms,
    )

    criticism_global = criticism_global_indices(
        df,
        criticisms,
    )

    coverage_metrics, criticism_df = (
        compute_explanation_metrics(
            df=df,
            prototypes=prototypes,
            criticism_positions=criticisms,
            scores=scores,
            witness=witness,
            feature_cols=feature_cols,
        )
    )

    print(
        "Criticism global indices:",
        criticism_global,
    )

    save_results(
        f"{name}_{output_suffix}",
        df,
        prototypes,
        criticisms,
        scores,
        witness,
    )

    return {
        "model": model,
        "prototypes": prototypes,
        "criticisms": criticisms,
        "scores": scores,
        "witness": witness,
        "coverage_metrics": coverage_metrics,
        "criticism_df": criticism_df,
    }

def compute_explanation_metrics(
    df: pd.DataFrame,
    prototypes,
    criticism_positions,
    scores,
    witness,
    feature_cols: list[str],
    radius: float = 0.10,
    k_neighbors: int = 20,
):
    """
    Analyze:
    1. prototype coverage
    2. criticism location
    3. criticism -> nearest prototype distance
    4. local/minority characteristics
    """

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy()

    n = len(X)

    # --------------------------------------------------------
    # Prototype positions
    # --------------------------------------------------------

    proto_positions = prototype_position_indices(
        df,
        prototypes,
    )

    proto_X = X[proto_positions]

    # --------------------------------------------------------
    # Distance from every sample -> nearest prototype
    # --------------------------------------------------------

    distance_matrix = pairwise_distances(
        X,
        proto_X,
        metric="euclidean",
    )

    nearest_proto_distance = (
        distance_matrix.min(axis=1)
    )

    nearest_proto_local_position = (
        distance_matrix.argmin(axis=1)
    )

    nearest_proto_position = np.asarray(
        proto_positions
    )[nearest_proto_local_position]

    nearest_proto_global_index = [
        df.index[i]
        for i in nearest_proto_position
    ]

    # --------------------------------------------------------
    # 1. Prototype coverage
    # --------------------------------------------------------

    coverage = {
        "coverage_mean_distance":
            float(
                nearest_proto_distance.mean()
            ),

        "coverage_median_distance":
            float(
                np.median(
                    nearest_proto_distance
                )
            ),

        "coverage_p90_distance":
            float(
                np.quantile(
                    nearest_proto_distance,
                    0.90,
                )
            ),

        "coverage_fraction":
            float(
                np.mean(
                    nearest_proto_distance <= radius
                )
            ),
    }

    # --------------------------------------------------------
    # 2. Local density
    # --------------------------------------------------------

    local_density = np.zeros(n)

    for cls in np.unique(y):

        cls_idx = np.where(y == cls)[0]

        if len(cls_idx) <= 1:
            local_density[cls_idx] = 0.0
            continue

        k = min(
            k_neighbors,
            len(cls_idx) - 1,
        )

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric="euclidean",
        )

        nn.fit(X[cls_idx])

        distances, _ = nn.kneighbors(
            X[cls_idx]
        )

        # remove self-distance
        mean_knn_distance = (
            distances[:, 1:].mean(axis=1)
        )

        # Larger = denser
        local_density[cls_idx] = (
            1.0
            / (mean_knn_distance + 1e-12)
        )

    # Percentile: 0 = sparsest, 100 = densest
    density_percentile = (
        np.argsort(
            np.argsort(local_density)
        )
        / max(n - 1, 1)
        * 100.0
    )

    # --------------------------------------------------------
    # Class frequency
    # --------------------------------------------------------

    class_counts = {
        cls: int(np.sum(y == cls))
        for cls in np.unique(y)
    }

    class_fraction = {
        cls: count / n
        for cls, count in class_counts.items()
    }

    minority_class = min(
        class_fraction,
        key=class_fraction.get,
    )

    # --------------------------------------------------------
    # 3 + 4. Criticism-specific information
    # --------------------------------------------------------

    criticism_rows = []

    for j, pos in enumerate(
        criticism_positions
    ):

        global_index = df.index[pos]

        nearest_pos = (
            nearest_proto_position[pos]
        )

        nearest_global = (
            nearest_proto_global_index[pos]
        )

        criticism_rows.append({
            "position": int(pos),
            "global_index": global_index,

            "t": df.iloc[pos]["t"],
            "y": df.iloc[pos]["y"],

            "x1": X[pos, 0],
            "x2": (
                X[pos, 1]
                if X.shape[1] > 1
                else np.nan
            ),

            "mmd_score": float(scores[j]),
            "witness": float(witness[j]),

            # ---- criticism -> nearest prototype
            "nearest_prototype_distance":
                float(
                    nearest_proto_distance[pos]
                ),

            "nearest_prototype_global_index":
                nearest_global,

            # # ---- coverage percentile
            # "coverage_distance_percentile":
            #     float(
            #         density_percentile[pos]
            #     ),

            # ---- local density
            "local_density":
                float(
                    local_density[pos]
                ),

            "local_density_percentile":
                float(
                    density_percentile[pos]
                ),

            # ---- class information
            "class_fraction":
                float(
                    class_fraction[
                        y[pos]
                    ]
                ),

            "is_minority_class":
                bool(
                    y[pos] == minority_class
                ),

            # Bottom 10% density
            "is_low_density_10pct":
                bool(
                    density_percentile[pos] <= 10
                ),

            # Bottom 20% density
            "is_low_density_20pct":
                bool(
                    density_percentile[pos] <= 20
                ),
        })

    criticism_df = pd.DataFrame(
        criticism_rows
    )

    # --------------------------------------------------------
    # 5. Aggregate criticism metrics
    # --------------------------------------------------------

    coverage["criticism_mean_distance"] = (
        float(
            criticism_df[
                "nearest_prototype_distance"
            ].mean()
        )
    )

    coverage["criticism_median_distance"] = (
        float(
            criticism_df[
                "nearest_prototype_distance"
            ].median()
        )
    )

    coverage["criticism_distance_ratio"] = (
        coverage["criticism_mean_distance"]
        / max(
            coverage[
                "coverage_mean_distance"
            ],
            1e-12,
        )
    )

    coverage["criticism_low_density_10pct"] = (
        float(
            criticism_df[
                "is_low_density_10pct"
            ].mean()
        )
    )

    coverage["criticism_low_density_20pct"] = (
        float(
            criticism_df[
                "is_low_density_20pct"
            ].mean()
        )
    )

    coverage["criticism_minority_fraction"] = (
        float(
            criticism_df[
                "is_minority_class"
            ].mean()
        )
    )

    return coverage, criticism_df

def summarize_pre_post(pre_rows, post_rows):

    pre = pd.DataFrame(pre_rows)
    post = pd.DataFrame(post_rows)

    metrics = [
        "coverage_mean_distance",
        "coverage_p90_distance",
        "criticism_mean_distance",
        "criticism_distance_ratio",
        "criticism_low_density_10pct",
        "criticism_low_density_20pct",
        "criticism_minority_fraction",
    ]

    rows = []

    for metric in metrics:

        delta = (
            post[metric]
            - pre[metric]
        )

        rows.append({
            "metric": metric,

            "pre_mean":
                pre[metric].mean(),

            "pre_std":
                pre[metric].std(),

            "post_mean":
                post[metric].mean(),

            "post_std":
                post[metric].std(),

            "delta_mean":
                delta.mean(),

            "delta_std":
                delta.std(),
        })

    return pd.DataFrame(rows)

def make_summary_row(stream, repeat, seed, phase, result):
    return {
        "stream": stream,
        "repeat": repeat,
        "seed": seed,
        "phase": phase,

        "n_prototypes": len(
            prototype_global_indices(result["prototypes"])
        ),
        "n_criticisms": len(result["criticisms"]),

        "mean_abs_witness": (
            float(np.mean(np.abs(result["witness"])))
            if len(result["witness"]) else np.nan
        ),
        "max_abs_witness": (
            float(np.max(np.abs(result["witness"])))
            if len(result["witness"]) else np.nan
        ),

        **result["coverage_metrics"],
    }

# ============================================================
# 10. Main experiment
# ============================================================

def main():

    RESULTS_DIR.mkdir(exist_ok=True)

    summary_rows = []

    for repeat in range(1, N_REPEATS + 1):

        seed = SEED_BASE + repeat

        print(
            f"\n{'=' * 70}"
        )
        print(
            f"REPEAT {repeat}/{N_REPEATS} | seed={seed}"
        )
        print(
            f"{'=' * 70}"
        )

        # ----------------------------------------------------
        # Generate one independent Sine stream
        # ----------------------------------------------------

        df = generate_sine(seed)

        pre_df, post_df = split_pre_post(df)

        print(
            f"Pre-drift : {len(pre_df)} samples "
            f"[{pre_df['t'].min()}, {pre_df['t'].max()}]"
        )

        print(
            f"Post-drift: {len(post_df)} samples "
            f"[{post_df['t'].min()}, {post_df['t'].max()}]"
        )

        # ----------------------------------------------------
        # PRE
        # ----------------------------------------------------
        pre_result = run_window(
            "pre",
            pre_df,
            seed,
            f"repeat{repeat:02d}",
        )

        # ----------------------------------------------------
        # POST
        # ----------------------------------------------------
        post_result = run_window(
            "post",
            post_df,
            seed,
            f"repeat{repeat:02d}",
        )

        # ----------------------------------------------------
        # Summary for this repeat
        # ----------------------------------------------------
        summary_rows.append(
            make_summary_row("sine", repeat, seed, "pre", pre_result)
        )

        summary_rows.append(
            make_summary_row("sine", repeat, seed, "post", post_result)
        )

        print(
            f"\nRepeat {repeat} finished."
        )

    # ========================================================
    # Save 10-repeat summary
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        RESULTS_DIR
        / "mmd_critic_sine_repeats_run001.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    pre_rows = summary[
        summary["phase"] == "pre"
    ].to_dict("records")

    post_rows = summary[
        summary["phase"] == "post"
    ].to_dict("records")

    pre_post_summary = summarize_pre_post(
        pre_rows,
        post_rows,
    )

    pre_post_path = (
        RESULTS_DIR
        / "mmd_critic_sine_pre_post_analysis_run001.csv"
    )

    pre_post_summary.to_csv(
        pre_post_path,
        index=False,
    )

    print(
        f"Saved pre-post analysis: {pre_post_path}"
    )

    print(
        "\n=== PRE vs POST ==="
    )

    print(
        pre_post_summary.to_string(index=False)
    )

    print(
        f"\nSaved summary: {summary_path}"
    )

    print(
        "\n=== 10-REPEAT SUMMARY ==="
    )

    print(
        summary.to_string(index=False)
    )

    print(
        "\n=== AGGREGATE BY PHASE ==="
    )

    for phase in ["pre", "post"]:
        phase_df = summary[summary["phase"] == phase]

        print(f"\n[{phase.upper()}]")

        for col in [
            "n_prototypes",
            "n_criticisms",
            "mean_abs_witness",
            "max_abs_witness",
            "coverage_mean_distance",
            "coverage_median_distance",
            "coverage_p90_distance",
            "coverage_fraction",
            "criticism_mean_distance",
            "criticism_median_distance",
            "criticism_distance_ratio",
            "criticism_low_density_10pct",
            "criticism_low_density_20pct",
            "criticism_minority_fraction",
        ]:
            print(
                f"{col}: "
                f"mean={phase_df[col].mean():.6f}, "
                f"std={phase_df[col].std():.6f}"
            )


if __name__ == "__main__":
    main()