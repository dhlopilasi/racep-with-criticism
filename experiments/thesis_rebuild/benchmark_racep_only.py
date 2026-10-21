"""Benchmark RACE-P without criticism or external drift detectors.

Run from the Protree repository root:
    python -m experiments.thesis_rebuild.benchmark_racep_only \
        --seed-start 2001 --n-seeds 1 \
        --output results/racep_only_timing_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import random
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from river import forest

from protree.detectors import RaceP
from protree.explainers import APete
from experiments.thesis_rebuild.controlled_streams import (
    Scenario,
    generate_controlled_stream,
)


def build_smoke_scenarios(chunk_size: int) -> list[Scenario]:
    """Same five controlled scenario templates as the existing smoke profile."""
    drift_start = 20 * chunk_size
    transition_width = 3 * chunk_size

    return [
        Scenario(
            scope="local",
            drift_kind="concept",
            schedule="abrupt",
            extent=0.10,
            magnitude="medium",
            drift_start=drift_start,
            transition_width=transition_width,
        ),
        Scenario(
            scope="global",
            drift_kind="concept",
            schedule="abrupt",
            extent=1.0,
            magnitude="medium",
            drift_start=drift_start,
            transition_width=transition_width,
        ),
        Scenario(
            scope="mixed",
            drift_kind="concept",
            schedule="abrupt",
            extent=0.10,
            magnitude="medium",
            drift_start=drift_start,
            transition_width=transition_width,
        ),
        Scenario(
            scope="local",
            drift_kind="virtual",
            schedule="abrupt",
            extent=0.10,
            magnitude="medium",
            drift_start=drift_start,
            transition_width=transition_width,
        ),
        Scenario(
            scope="global",
            drift_kind="stationary",
            schedule="abrupt",
            extent=1.0,
            magnitude="medium",
            drift_start=drift_start,
            transition_width=transition_width,
        ),
    ]


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def count_prototypes(prototype_dict) -> int:
    if not isinstance(prototype_dict, dict):
        return 0
    return sum(len(values) for values in prototype_dict.values())


def ground_truth_class(
    block: pd.DataFrame,
    scenario: Scenario,
) -> str:
    has_concept_drift = (
        bool(block["gt_concept_drift"].any())
        if "gt_concept_drift" in block.columns
        else False
    )
    if not has_concept_drift:
        return "none"

    if scenario.scope == "global":
        return "global"
    if scenario.scope == "local":
        return "local"
    if scenario.scope == "mixed":
        return "drift"
    return "none"


def run_racep_only(
    frame: pd.DataFrame,
    scenario: Scenario,
    seed: int,
    chunk_size: int,
    reference_chunks: int,
) -> list[dict]:
    """Run only the model update and the native RACE-P detector."""
    random.seed(seed)
    np.random.seed(seed)

    feature_cols = ["x1", "x2"]
    arf_params = inspect.signature(forest.ARFClassifier).parameters
    model_kwargs = {"seed": seed} if "seed" in arf_params else {}

    model = forest.ARFClassifier(**model_kwargs)

    detector = RaceP(
        model=model,
        prototype_selector=APete,
        prototype_selector_kwargs={"alpha": 0.01},
        measure="prototype_reassignment_impact",
        strategy="total",
        grace_period=2 * reference_chunks,
        const=3.0,
    )

    n_chunks = len(frame) // chunk_size
    rows = []

    scenario_id = (
        f"{scenario.scope}_{scenario.drift_kind}_{scenario.schedule}_"
        f"e{scenario.extent:g}_{scenario.magnitude}_"
        f"cs{chunk_size}_seed{seed}"
    )

    for chunk_id in range(n_chunks):
        chunk_started = time.perf_counter()

        # Prepare only the inputs required by RACE-P.
        preparation_started = time.perf_counter()
        start = chunk_id * chunk_size
        end = start + chunk_size
        block = frame.iloc[start:end].copy()

        X = block[feature_cols].to_numpy(dtype=float)
        y = block["y"].to_numpy(dtype=int)
        x_block = block[feature_cols].to_dict(orient="records")
        y_block = y.tolist()

        preparation_ms = (
            time.perf_counter() - preparation_started
        ) * 1000.0

        # Match the model-update order in the official Protree README:
        # learn the current block, then call detector.update().
        model_started = time.perf_counter()
        for x_i, y_i in zip(x_block, y_block):
            model.learn_one(x_i, y_i)

        model_update_ms = (
            time.perf_counter() - model_started
        ) * 1000.0

        detector_started = time.perf_counter()
        detector.update(x_block, y_block)
        detector_update_ms = (
            time.perf_counter() - detector_started
        ) * 1000.0

        chunk_wall_ms = (
            time.perf_counter() - chunk_started
        ) * 1000.0

        raw_metric = detector.metric_value
        if isinstance(raw_metric, dict):
            raise TypeError(
                "Expected scalar metric_value with strategy='total'."
            )
        p_signal = (
            float(raw_metric)
            if raw_metric is not None and np.isscalar(raw_metric)
            else np.nan
        )

        native_alarm = bool(detector.drift_detected)
        scored_alarm = (
            native_alarm and chunk_id >= 2 * reference_chunks
        )

        prototypes_previous, prototypes_current = detector.prototypes
        gt_class = ground_truth_class(block, scenario)

        row = {
            "seed": seed,
            "chunk": chunk_id,
            "t_start": start,
            "t_end": end - 1,
            "scenario_id": scenario_id,
            "stream_scope": scenario.scope,
            "drift_kind": scenario.drift_kind,
            "schedule": scenario.schedule,
            "extent": scenario.extent,
            "magnitude": scenario.magnitude,
            "ground_truth_class": gt_class,
            "gt_concept_drift": (
                bool(block["gt_concept_drift"].any())
                if "gt_concept_drift" in block.columns
                else False
            ),
            "gt_distribution_shift": (
                bool(block["gt_distribution_shift"].any())
                if "gt_distribution_shift" in block.columns
                else False
            ),
            "n_prototypes_previous": count_prototypes(
                prototypes_previous
            ),
            "n_prototypes_current": count_prototypes(
                prototypes_current
            ),
            "p_pri": p_signal,
            "racep_native_alarm_raw": native_alarm,
            "racep_native_alarm_scored": scored_alarm,
            "chunk_preparation_ms": preparation_ms,
            "racep_model_update_ms": model_update_ms,
            "racep_update_ms": detector_update_ms,
            "racep_component_sum_ms": (
                model_update_ms + detector_update_ms
            ),
            "racep_chunk_wall_ms": chunk_wall_ms,
        }
        rows.append(row)

    return rows


def summarize_scenarios(
    raw: pd.DataFrame,
    reference_chunks: int,
) -> pd.DataFrame:
    evaluated = raw[raw["chunk"] >= 2 * reference_chunks]
    results = []

    for scenario_id, group in evaluated.groupby("scenario_id"):
        labels = group["ground_truth_class"].dropna().unique()
        if len(labels) != 1:
            raise ValueError(
                f"Inconsistent ground truth for {scenario_id}: {labels}"
            )

        gt_class = str(labels[0])
        alarms = group.loc[
            group["racep_native_alarm_scored"], "chunk"
        ].astype(int).tolist()

        first_alarm = min(alarms) if alarms else np.nan
        event_start_chunk = 2 * reference_chunks

        delay = (
            float(first_alarm - event_start_chunk)
            if alarms and gt_class != "none"
            else np.nan
        )

        results.append({
            "seed": int(group["seed"].iloc[0]),
            "scenario_id": scenario_id,
            "stream_scope": group["stream_scope"].iloc[0],
            "drift_kind": group["drift_kind"].iloc[0],
            "ground_truth_class": gt_class,
            "n_evaluated_chunks": len(group),
            "n_alarm_chunks": len(alarms),
            "has_alarm": bool(alarms),
            "alarm_chunks": ",".join(map(str, alarms)),
            "first_alarm_chunk": first_alarm,
            "delay_chunks": delay,
        })

    return pd.DataFrame(results)


def summarize_timings(raw: pd.DataFrame, reference_chunks: int) -> pd.DataFrame:
    timing_columns = [
        "chunk_preparation_ms",
        "racep_model_update_ms",
        "racep_update_ms",
        "racep_component_sum_ms",
        "racep_chunk_wall_ms",
    ]

    subsets = [
        ("all_chunks", raw),
        ("evaluated_chunks", raw[raw["chunk"] >= 2 * reference_chunks]),
    ]

    results = []
    for phase, df in subsets:
        for column in timing_columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            results.append({
                "phase": phase,
                "timing_metric": column,
                "n_chunks": len(values),
                "mean_ms": float(values.mean()),
                "median_ms": float(values.median()),
                "p95_ms": float(values.quantile(0.95)),
                "max_ms": float(values.max()),
            })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RACE-P without criticism or external detectors."
    )
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--n-chunks", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--reference-chunks", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/racep_only_timing_smoke"),
    )
    args = parser.parse_args()

    if args.n_seeds < 1:
        parser.error("--n-seeds must be at least 1")
    if args.n_chunks <= 2 * args.reference_chunks:
        parser.error(
            "n_chunks must exceed 2 * reference_chunks to leave "
            "chunks for evaluation."
        )
    if 20 < 2 * args.reference_chunks or 20 >= args.n_chunks:
        parser.error(
            "This benchmark keeps the smoke-profile drift start at chunk 20; "
            "ensure 20 >= 2*reference_chunks and 20 < n_chunks."
        )

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)

    if any(outdir.glob("racep_only_*.csv")) or (
        outdir / "racep_only_manifest.json"
    ).exists():
        raise FileExistsError(
            f"Output directory already contains benchmark results: {outdir}. "
            "Choose a new output directory to avoid overwriting data."
        )

    scenarios = build_smoke_scenarios(args.chunk_size)
    all_raw = []

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        seed_rows = []

        for index, scenario in enumerate(scenarios, start=1):
            print(
                f"[seed={seed}] scenario {index}/{len(scenarios)}: "
                f"{scenario.scope}/{scenario.drift_kind}",
                flush=True,
            )

            # Stream generation occurs outside the measured RACE-P timings.
            frame = generate_controlled_stream(
                args.n_chunks * args.chunk_size,
                seed,
                scenario,
            )

            rows = run_racep_only(
                frame=frame,
                scenario=scenario,
                seed=seed,
                chunk_size=args.chunk_size,
                reference_chunks=args.reference_chunks,
            )
            seed_rows.extend(rows)

        seed_df = pd.DataFrame(seed_rows)
        seed_summary = summarize_scenarios(
            seed_df, args.reference_chunks
        )

        raw_seed_path = outdir / f"racep_only_raw_seed{seed}.csv"
        summary_seed_path = (
            outdir / f"racep_only_scenario_summary_seed{seed}.csv"
        )
        seed_df.to_csv(raw_seed_path, index=False)
        seed_summary.to_csv(summary_seed_path, index=False)

        all_raw.append(seed_df)
        print(f"Saved seed output: {raw_seed_path}", flush=True)

    raw = pd.concat(all_raw, ignore_index=True)
    scenarios_summary = summarize_scenarios(
        raw, args.reference_chunks
    )
    grouped_summary = (
        scenarios_summary.groupby(
            ["stream_scope", "drift_kind", "ground_truth_class"],
            dropna=False,
        )
        .agg(
            n_runs=("seed", "nunique"),
            runs_with_alarm=("has_alarm", "sum"),
            scenario_alarm_rate=("has_alarm", "mean"),
            total_alarm_chunks=("n_alarm_chunks", "sum"),
            mean_delay_chunks_when_detected=("delay_chunks", "mean"),
        )
        .reset_index()
    )
    timing_summary = summarize_timings(
        raw, args.reference_chunks
    )

    raw.to_csv(outdir / "racep_only_raw.csv", index=False)
    scenarios_summary.to_csv(
        outdir / "racep_only_scenario_summary.csv", index=False
    )
    grouped_summary.to_csv(
        outdir / "racep_only_grouped_summary.csv", index=False
    )
    timing_summary.to_csv(
        outdir / "racep_only_timing_summary.csv", index=False
    )

    source_path = Path(__file__).resolve()
    script_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    manifest = {
        "benchmark": "RACE-P only; no criticism or external drift detectors",
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "seeds": list(
            range(args.seed_start, args.seed_start + args.n_seeds)
        ),
        "n_chunks": args.n_chunks,
        "chunk_size": args.chunk_size,
        "reference_chunks": args.reference_chunks,
        "drift_start_chunk": 20,
        "prototype_selector": "APete",
        "prototype_selector_alpha": 0.01,
        "measure": "prototype_reassignment_impact",
        "strategy": "total",
        "grace_period": 2 * args.reference_chunks,
        "const": 3.0,
        "model_update_order": "learn current chunk, then detector.update",
        "timings_exclude": [
            "stream generation",
            "model and detector initialization",
            "criticism",
            "ADWIN",
            "Page-Hinkley",
            "separate error-monitoring model",
            "CSV writing",
        ],
        "git_head": None,
        "benchmark_script_sha256": script_hash,
        "software_versions": {
            "python": platform.python_version(),
            "river": package_version("river"),
            "protree": package_version("protree"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "note": (
            "A controlled five-scenario benchmark for this thesis; "
            "not a claim of reproducing every experiment in the paper."
        ),
    }

    try:
        import subprocess
        manifest["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        pass

    (outdir / "racep_only_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== RACE-P-ONLY SCENARIO SUMMARY ===")
    print(grouped_summary.round(4).to_string(index=False))

    print("\n=== RACE-P-ONLY TIMING SUMMARY ===")
    print(timing_summary.round(3).to_string(index=False))

    print(f"\nSaved benchmark outputs under: {outdir}")


if __name__ == "__main__":
    main()
