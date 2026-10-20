from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_DIR = Path("results/thesis_rebuild_e2_e4")

RUNS = {
    1001: "controls5_ref10_v2_seed1001",
    1002: "controls5_ref10_fixed_seed1002",
    1003: "controls5_ref10_v2_seed1003",
    1004: "controls5_ref10_v2_seed1004",
    1005: "controls5_ref10_v2_seed1005",
}

EXPECTED_ROWS = {
    "raw_signals": 150,
    "detection_metrics": 855,
    "four_zone_chunks": 3600,
}

ROWS_PER_SCENARIO = {
    "raw_signals": 30,
    "detection_metrics": 171,
    "four_zone_chunks": 720,
}


# ------------------------------------------------------------
# 1. Load and validate the five runs
# ------------------------------------------------------------

parts = {
    name: []
    for name in EXPECTED_ROWS
}

for seed, run_id in RUNS.items():
    print(f"Checking seed {seed}: {run_id}")

    for prefix in EXPECTED_ROWS:
        path = DATA_DIR / f"{prefix}_{run_id}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        if len(df) != EXPECTED_ROWS[prefix]:
            raise ValueError(
                f"{path.name}: expected {EXPECTED_ROWS[prefix]} "
                f"rows, found {len(df)}"
            )

        if "seed" not in df.columns:
            raise ValueError(f"{path.name}: missing seed column")

        seeds_found = set(df["seed"].dropna().astype(int).unique())

        if seeds_found != {seed}:
            raise ValueError(
                f"{path.name}: expected seed {seed}, "
                f"found {seeds_found}"
            )

        if "scenario_id" in df.columns:
            counts = df.groupby("scenario_id").size()

            if len(counts) != 5:
                raise ValueError(
                    f"{path.name}: expected 5 scenarios, "
                    f"found {len(counts)}"
                )

            expected = ROWS_PER_SCENARIO[prefix]

            if not (counts == expected).all():
                raise ValueError(
                    f"{path.name}: unexpected rows per scenario:\n"
                    f"{counts.to_string()}"
                )

        parts[prefix].append(df)

        print(
            f"  {prefix}: {len(df)} rows, "
            f"{df['scenario_id'].nunique()} scenarios"
        )


# ------------------------------------------------------------
# 2. Combine detection metrics across seeds
# ------------------------------------------------------------

detection = pd.concat(
    parts["detection_metrics"],
    ignore_index=True,
)

detection.to_csv(
    DATA_DIR / "multiseed_detection_metrics_rows.csv",
    index=False,
)

# Do not group by seed-specific scenario_id or run_id.
# Keep the experimental configuration and detector separate.
detection_keys = [
    "scope",
    "drift_kind",
    "schedule",
    "extent",
    "magnitude",
    "chunk_size",
    "k",
    "w",
    "c_signal",
    "rule",
    "baseline_parameters",
]

detection_metrics_to_summarize = [
    "tp",
    "fn",
    "fp",
    "precision",
    "recall",
    "f1",
    "mean_delay_chunks",
    "fp_per_1000_chunks",
    "n_alarms",
    "n_true_events",
    "four_zone_chunk_accuracy",
    "four_zone_balanced_accuracy_supported_classes",
]

detection_metrics_to_summarize = [
    column
    for column in detection_metrics_to_summarize
    if column in detection.columns
]

summary_rows = []

for key, group in detection.groupby(
    detection_keys,
    dropna=False,
    sort=False,
):
    if not isinstance(key, tuple):
        key = (key,)

    row = dict(zip(detection_keys, key))
    row["n_rows"] = len(group)
    row["n_seeds"] = group["seed"].nunique()

    for metric in detection_metrics_to_summarize:
        values = pd.to_numeric(group[metric], errors="coerce")

        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=1)

    summary_rows.append(row)

detection_summary = pd.DataFrame(summary_rows)

detection_summary.to_csv(
    DATA_DIR / "multiseed_detection_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 3. Combine four-zone predictions
# ------------------------------------------------------------

zones = pd.concat(
    parts["four_zone_chunks"],
    ignore_index=True,
)

# The controlled drift starts at chunk 20.
# Evaluate the post-start period, including stationary and
# virtual-drift controls as negative controls for concept drift.
zones_post = zones.loc[zones["chunk"] >= 20].copy()

# Bỏ phần seed ở cuối scenario_id để ghép cùng kịch bản
# giữa các seed.
zones_post["scenario"] = zones_post["scenario_id"].str.replace(
    r"_seed\d+$",
    "",
    regex=True,
)

zones_post.to_csv(
    DATA_DIR / "multiseed_four_zone_post_rows.csv",
    index=False,
)

# ------------------------------------------------------------
# 4. Four-zone metrics per seed, k, and C-signal
# ------------------------------------------------------------

per_seed_rows = []

for (seed, k, c_signal), group in zones_post.groupby(
    ["seed", "k", "c_signal"],
    dropna=False,
):
    truth = group["ground_truth_class"].astype(str)
    prediction = group["zone_prediction"].astype(str)

    accuracy = float((truth == prediction).mean())

    # Balanced accuracy: mean recall across classes present
    # in the ground truth for this group.
    recalls = []

    for class_name in truth.unique():
        mask = truth == class_name
        class_recall = float(
            (prediction[mask] == class_name).mean()
        )
        recalls.append(class_recall)

    balanced_accuracy = (
        float(np.mean(recalls))
        if recalls
        else np.nan
    )

    per_seed_rows.append(
        {
            "seed": int(seed),
            "k": k,
            "c_signal": c_signal,
            "n_chunks": len(group),
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
        }
    )

per_seed = pd.DataFrame(per_seed_rows)

per_seed.to_csv(
    DATA_DIR / "multiseed_four_zone_per_seed_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 5. Mean and standard deviation across seeds
# ------------------------------------------------------------

zone_summary = (
    per_seed.groupby(["k", "c_signal"], dropna=False)
    .agg(
        n_seeds=("seed", "nunique"),
        n_chunks_per_seed=("n_chunks", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
    )
    .reset_index()
)

zone_summary.to_csv(
    DATA_DIR / "multiseed_four_zone_summary.csv",
    index=False,
)


# ------------------------------------------------------------
# 6. Confusion counts across all five seeds
# ------------------------------------------------------------

confusion = (
    zones_post.groupby(
        [
            "k",
            "c_signal",
            "ground_truth_class",
            "zone_prediction",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
)

confusion.to_csv(
    DATA_DIR / "multiseed_four_zone_confusion_counts.csv",
    index=False,
)

# ------------------------------------------------------------
# 6B. Four-zone evaluation by individual scenario
# ------------------------------------------------------------

scenario_keys = [
    "seed",
    "scenario",
    "ground_truth_class",
    "k",
    "c_signal",
]

# Mỗi kịch bản phải có một ground-truth class cố định
# trong giai đoạn hậu drift.
gt_counts = zones_post.groupby(
    ["seed", "scenario"]
)["ground_truth_class"].nunique()

if (gt_counts != 1).any():
    raise ValueError(
        "Ground-truth class varies within a scenario."
    )

zones_post["is_correct"] = (
    zones_post["zone_prediction"].astype(str)
    == zones_post["ground_truth_class"].astype(str)
)

# Tính accuracy riêng cho mỗi seed và kịch bản.
scenario_per_seed = (
    zones_post.groupby(scenario_keys, dropna=False)
    .agg(
        n_chunks=("chunk", "size"),
        accuracy=("is_correct", "mean"),
    )
    .reset_index()
)

scenario_per_seed.to_csv(
    DATA_DIR / "multiseed_four_zone_per_scenario_per_seed.csv",
    index=False,
)

# Tổng hợp qua năm seed, nhưng không gộp các kịch bản.
scenario_summary = (
    scenario_per_seed.groupby(
        [
            "scenario",
            "ground_truth_class",
            "k",
            "c_signal",
        ],
        dropna=False,
    )
    .agg(
        n_seeds=("seed", "nunique"),
        n_chunks_per_seed=("n_chunks", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    .reset_index()
)

scenario_summary.to_csv(
    DATA_DIR / "multiseed_four_zone_per_scenario_summary.csv",
    index=False,
)

# Ma trận nhầm lẫn được tách theo từng kịch bản.
scenario_confusion = (
    zones_post.groupby(
        [
            "scenario",
            "k",
            "c_signal",
            "ground_truth_class",
            "zone_prediction",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
)

scenario_confusion.to_csv(
    DATA_DIR / "multiseed_four_zone_per_scenario_confusion.csv",
    index=False,
)

print("\nFour-zone accuracy by scenario:")
print(
    scenario_summary.sort_values(
        ["scenario", "k", "c_signal"]
    ).to_string(index=False)
)

# ------------------------------------------------------------
# 7. Print key outputs
# ------------------------------------------------------------

print("\nAll five runs passed validation.")

print("\nCombined row counts:")
print(f"  detection_metrics: {len(detection)}")
print(f"  four_zone_chunks, all chunks: {len(zones)}")
print(f"  four_zone_chunks, chunk >= 20: {len(zones_post)}")

print("\nDetection summary preview:")
print(
    detection_summary[
        [
            "scope",
            "drift_kind",
            "k",
            "w",
            "c_signal",
            "rule",
            "n_seeds",
            "f1_mean",
            "f1_std",
            "mean_delay_chunks_mean",
            "fp_per_1000_chunks_mean",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nFour-zone summary:")
print(
    zone_summary
    .sort_values(["k", "c_signal"])
    .to_string(index=False)
)

print("\nSaved analysis files under:")
print(DATA_DIR)