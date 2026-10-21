"""Run controlled-stream E2/E4 pilot for RACE-P + criticism.

Run from the Protree repository root, after copying this bundle into the repo:
    python -m experiments.thesis_rebuild.run_controlled_e2_e4 --profile smoke

The script records raw per-chunk P/C signals and then applies independently
calibrated thresholds to P and each candidate C signal. It does not alter RACE-P
code; it reads RaceP.metric_value after the official detector update.
"""
from __future__ import annotations

import argparse
import inspect
import json
import random
import subprocess
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from experiments.thesis_rebuild.controlled_streams import Scenario, generate_controlled_stream
from experiments.thesis_rebuild.cross_window_mmd_critic import CrossWindowMMDCritic, resolve_rbf_gamma
from experiments.thesis_rebuild.evaluation import (
    above_threshold,
    combine_p_c,
    event_detection_metrics,
    fit_upper_threshold,
    four_zone,
    four_zone_metrics,
)


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _as_frame(points, feature_cols):
    if points is None:
        return pd.DataFrame(columns=feature_cols)
    if isinstance(points, pd.DataFrame):
        frame = points.copy()
    elif isinstance(points, pd.Series):
        frame = points.to_frame().T
    elif isinstance(points, dict):
        try:
            frame = pd.DataFrame(points)
        except ValueError:
            frame = pd.DataFrame([points])
        if frame.shape[1] == len(feature_cols) and not set(feature_cols).issubset(frame.columns):
            frame.columns = feature_cols
    elif isinstance(points, np.ndarray):
        arr = np.asarray(points)
        frame = pd.DataFrame(arr, columns=feature_cols if arr.ndim == 2 and arr.shape[1] == len(feature_cols) else None)
    else:
        frame = pd.DataFrame(list(points))
        if frame.shape[1] == len(feature_cols) and not set(feature_cols).issubset(frame.columns):
            frame.columns = feature_cols
    if len(frame) == 0:
        return frame
    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Prototype data missing feature columns: {missing}; columns={list(frame.columns)}")
    return frame


def flatten_prototypes(prototype_dict, feature_cols, source_X):
    """Return prototype features, class labels, and local source positions.

    A-PETE returns actual instances grouped by class. Positions are recovered by
    matching those instances back to the source chunk, which is robust to DataFrame
    index resets in adapters. Labels are the corresponding class dictionary keys.
    """
    xs, ys, positions = [], [], []
    source_X = np.asarray(source_X, dtype=float)
    if not prototype_dict:
        return np.empty((0, source_X.shape[1])), np.empty(0), np.empty(0, dtype=int)

    for class_label, points in prototype_dict.items():
        frame = _as_frame(points, feature_cols)
        if frame.empty:
            continue
        arr = frame[feature_cols].to_numpy(dtype=float)
        if not len(arr):
            continue
        xs.append(arr)
        try:
            numeric_label = float(class_label)
        except (TypeError, ValueError):
            numeric_label = float(abs(hash(str(class_label))) % 100_000) / 100_000.0
        ys.extend([numeric_label] * len(arr))
        d = pairwise_distances(arr, source_X, metric="euclidean")
        positions.extend(np.argmin(d, axis=1).astype(int).tolist())

    if not xs:
        return np.empty((0, source_X.shape[1])), np.empty(0), np.empty(0, dtype=int)
    return np.vstack(xs), np.asarray(ys, dtype=float), np.asarray(positions, dtype=int)


def _nearest_prototype_labels(X, proto_X, proto_y):
    if len(proto_X) == 0:
        return np.full(len(X), np.nan)
    distances = pairwise_distances(np.asarray(X, dtype=float), proto_X, metric="euclidean")
    return proto_y[np.argmin(distances, axis=1)]


def _build_scenarios(profile: str, chunk_size: int, n_chunks_override: int | None, ref_chunks_override: int | None, seed_start: int, n_seeds: int):
    if profile == "smoke":
        n_chunks = n_chunks_override or 20
        ref_chunks = ref_chunks_override or 5
        drift_start = 20 * chunk_size
        transition_width = 3 * chunk_size

        scenarios = [
            # 1. Local concept drift
            (
                Scenario(
                    scope="local",
                    drift_kind="concept",
                    schedule="abrupt",
                    extent=0.10,
                    magnitude="medium",
                    drift_start=drift_start,
                    transition_width=transition_width,
                ),
                seed_start,
            ),

            # 2. Global concept drift
            (
                Scenario(
                    scope="global",
                    drift_kind="concept",
                    schedule="abrupt",
                    extent=1.0,
                    magnitude="medium",
                    drift_start=drift_start,
                    transition_width=transition_width,
                ),
                seed_start,
            ),

            # 3. Mixed concept drift
            (
                Scenario(
                    scope="mixed",
                    drift_kind="concept",
                    schedule="abrupt",
                    extent=0.10,
                    magnitude="medium",
                    drift_start=drift_start,
                    transition_width=transition_width,
                ),
                seed_start,
            ),

            # 4. Virtual drift: distribution changes, concept does not
            (
                Scenario(
                    scope="local",
                    drift_kind="virtual",
                    schedule="abrupt",
                    extent=0.10,
                    magnitude="medium",
                    drift_start=drift_start,
                    transition_width=transition_width,
                ),
                seed_start,
            ),

            # 5. Stationary control: no drift
            (
                Scenario(
                    scope="global",
                    drift_kind="stationary",
                    schedule="abrupt",
                    extent=1.0,
                    magnitude="medium",
                    drift_start=drift_start,
                    transition_width=transition_width,
                ),
                seed_start,
            ),
        ]

        return n_chunks, ref_chunks, scenarios

    if profile in {"pilot", "final"}:
        n_chunks = n_chunks_override or 40
        ref_chunks = ref_chunks_override or (8 if profile == "pilot" else 10)
        scenarios = []
        seeds = list(range(seed_start, seed_start + n_seeds))
        extents = (0.05, 0.25, 0.50) if profile == "pilot" else (0.05, 0.10, 0.25, 0.50)
        for extent in extents:
            for magnitude in ("small", "medium", "large"):
                for schedule in ("abrupt", "gradual"):
                    for seed in seeds:
                        scenarios.append((Scenario(
                            scope="local", drift_kind="concept", schedule=schedule,
                            extent=extent, magnitude=magnitude,
                            drift_start=20 * chunk_size,
                            transition_width=5 * chunk_size,
                        ), seed))
        for scope in ("global", "mixed"):
            for magnitude in ("small", "medium", "large"):
                for schedule in ("abrupt", "gradual"):
                    for seed in seeds:
                        scenarios.append((Scenario(
                            scope=scope, drift_kind="concept", schedule=schedule,
                            extent=1.0 if scope == "global" else 0.10,
                            magnitude=magnitude, drift_start=20 * chunk_size,
                            transition_width=5 * chunk_size,
                        ), seed))
        # Virtual drift and stationary controls are necessary to measure false alarms.
        virtual_cases = (("local", 0.10), ("local", 0.50), ("global", 1.0)) if profile == "pilot" else tuple(("local", e) for e in (0.05, 0.10, 0.25, 0.50)) + (("global", 1.0),)
        virtual_schedules = ("abrupt",) if profile == "pilot" else ("abrupt", "gradual")
        for scope, extent in virtual_cases:
            for magnitude in ("small", "medium", "large"):
                for schedule in virtual_schedules:
                    for seed in seeds:
                        scenarios.append((Scenario(
                            scope=scope, drift_kind="virtual", schedule=schedule,
                            extent=extent, magnitude=magnitude,
                            drift_start=20 * chunk_size, transition_width=5 * chunk_size,
                        ), seed))
        for seed in seeds:
            scenarios.append((Scenario(
                scope="global", drift_kind="stationary", schedule="abrupt", extent=1.0,
                magnitude="medium", drift_start=20 * chunk_size,
                transition_width=5 * chunk_size,
            ), seed))
        return n_chunks, ref_chunks, scenarios

    raise ValueError("profile must be smoke, pilot, or final")



def run_one_stream(
    frame: pd.DataFrame,
    scenario: Scenario,
    seed: int,
    chunk_size: int,
    ref_chunks: int,
    k_values,
    w_values,
    n_criticisms: int,
    event_tolerance: int,
    adwin_delta: float = 0.002,
    ph_delta: float = 0.005,
    ph_threshold: float = 50.0,
    ph_alpha: float = 0.9999,
    ph_min_instances: int = 30,
):
    # Keep the official RACE-P object and prototype selector as the source of P.
    # ADWIN and Page-Hinkley receive the same sample-wise prequential error sequence.
    from river import drift, forest
    from protree.detectors import RaceP
    from protree.explainers import APete

    random.seed(seed)
    np.random.seed(seed)

    feature_cols = ["x1", "x2"]
    arf_params = inspect.signature(forest.ARFClassifier).parameters
    model_kwargs = {"seed": seed} if "seed" in arf_params else {}

    error_model = forest.ARFClassifier(**model_kwargs)
    racep_model = forest.ARFClassifier(**model_kwargs)
    adwin = drift.ADWIN(delta=adwin_delta)
    ph_params = inspect.signature(drift.PageHinkley).parameters
    ph_kwargs = {
        "delta": ph_delta,
        "threshold": ph_threshold,
        "alpha": ph_alpha,
        "min_instances": ph_min_instances,
    }
    if "mode" in ph_params:
        ph_kwargs["mode"] = "up"  # focus on increases in error rate
    ph_kwargs = {key: value for key, value in ph_kwargs.items() if key in ph_params}
    page_hinkley = drift.PageHinkley(**ph_kwargs)
    detector = RaceP(
        model=racep_model,
        prototype_selector=APete,
        prototype_selector_kwargs={"alpha": 0.01},
        measure="prototype_reassignment_impact",
        strategy="total",
        grace_period=2 * ref_chunks,
        const=3.0,  # preserve the original RACE-P default detector threshold
    )

    first_Z = np.column_stack([frame.loc[: chunk_size - 1, feature_cols].to_numpy(), frame.loc[: chunk_size - 1, "y"].to_numpy()])
    gamma = resolve_rbf_gamma(first_Z, "scale")  # fixed once from the initial reference chunk
    critic = CrossWindowMMDCritic(
        n_criticisms=n_criticisms,
        gamma=gamma,
        max_candidates=512,
        max_current_reference=1024,
        diversity_weight=0.05,
    )

    raw_rows, detail_rows = [], []
    n_chunks = len(frame) // chunk_size
    racep_native_chunk_flags = np.zeros(n_chunks, dtype=bool)
    adwin_chunk_flags = np.zeros(n_chunks, dtype=bool)
    page_hinkley_chunk_flags = np.zeros(n_chunks, dtype=bool)
    adwin_chunk_count = np.zeros(n_chunks, dtype=int)
    page_hinkley_chunk_count = np.zeros(n_chunks, dtype=int)
    previous_block_X = None
    for chunk_id in range(n_chunks):
        total_started = time.perf_counter()
        start = chunk_id * chunk_size
        end = start + chunk_size
        block = frame.iloc[start:end].copy()
        X = block[feature_cols].to_numpy(dtype=float)
        y = block["y"].to_numpy(dtype=int)
        x_block = block[feature_cols].to_dict(orient="records")
        y_block = y.tolist()

        # Sample-wise prequential error: predict first, then learn this sample.
        # The same error sequence is then replayed in order to ADWIN and PH.
        model_started = time.perf_counter()
        errors = []
        for x_i, y_i in zip(x_block, y_block):
            pred = error_model.predict_one(x_i)
            error = float(pred != y_i) if pred is not None else np.nan
            errors.append(error)
            error_model.learn_one(x_i, y_i)
        model_update_ms = (time.perf_counter() - model_started) * 1000.0

        external_started = time.perf_counter()
        for error in errors:
            if not np.isfinite(error):
                continue
            adwin.update(float(error))
            if adwin.drift_detected:
                adwin_chunk_flags[chunk_id] = True
                adwin_chunk_count[chunk_id] += 1
            page_hinkley.update(float(error))
            if page_hinkley.drift_detected:
                page_hinkley_chunk_flags[chunk_id] = True
                page_hinkley_chunk_count[chunk_id] += 1
        external_detector_ms = (time.perf_counter() - external_started) * 1000.0
        
        # Match the official Protree usage order:
        # train the model on the current chunk before updating RACE-P.
        racep_model_update_started = time.perf_counter()

        for x_i, y_i in zip(x_block, y_block):
            racep_model.learn_one(x_i, y_i)

        racep_model_update_ms = (
            time.perf_counter() - racep_model_update_started
        ) * 1000.0

        # Update the original RACE-P detector.
        racep_started = time.perf_counter()

        detector.update(x_block, y_block)
        racep_native_chunk_flags[chunk_id] = bool(detector.drift_detected)

        racep_ms = (
            time.perf_counter() - racep_started
        ) * 1000.0

        p_signal = detector.metric_value
        if isinstance(p_signal, dict):
            # strategy='total' should produce scalar PRI; fail loudly if API changes.
            raise TypeError("Expected scalar RACE-P metric_value with strategy='total'.")
        p_signal = float(p_signal) if p_signal is not None and np.isscalar(p_signal) else np.nan

        previous_prototypes, current_prototypes = detector.prototypes
        prev_source_X = previous_block_X if previous_block_X is not None else X
        prev_X, prev_y, prev_positions = flatten_prototypes(previous_prototypes, feature_cols, prev_source_X)
        curr_X, curr_y, curr_positions = flatten_prototypes(current_prototypes, feature_cols, X)

        if chunk_id < 3:
            print(f"\n[DEBUG] chunk={chunk_id}")
            print("Class counts:", pd.Series(y).value_counts().to_dict())

            for name, proto_dict in [
                ("previous", previous_prototypes),
                ("current", current_prototypes),
            ]:
                print(
                    f"{name} raw type:",
                    type(proto_dict).__name__,
                )

                if isinstance(proto_dict, dict):
                    print("  class keys:", list(proto_dict.keys()))

                    for cls, points in proto_dict.items():
                        print(
                            f"  class={cls!r}, "
                            f"type={type(points).__name__}, "
                            f"shape={getattr(points, 'shape', None)}, "
                            f"len={len(points) if hasattr(points, '__len__') else 'N/A'}"
                        )
                else:
                    print("  raw value:", repr(proto_dict)[:500])

            print(
                "Flattened previous:",
                len(prev_X),
                "prototypes;",
                "labels=",
                prev_y.tolist(),
            )

            print(
                "Flattened current:",
                len(curr_X),
                "prototypes;",
                "labels=",
                curr_y.tolist(),
            )

            print("PRI metric_value:", detector.metric_value)
        c_witness = np.nan
        c_distance = np.nan
        c_reassignment = np.nan
        c_fraction = np.nan
        critic_positions = np.empty(0, dtype=int)
        critic_abs = np.empty(0)
        critic_witness = np.empty(0)
        random_witness = np.nan
        error_witness = np.nan
        ref_region_fraction = np.nan
        curr_proto_region_fraction = np.nan
        random_region_fraction = np.nan
        error_region_fraction = np.nan
        n_error_criticism = 0
        criticism_ms = 0.0

        if chunk_id >= 1 and len(prev_X):
            criticism_started = time.perf_counter()
            Z_current = np.column_stack([X, y.astype(float)])
            Z_prev_proto = np.column_stack([prev_X, prev_y])
            selected = critic.select_criticisms(Z_current, Z_prev_proto, seed=seed + chunk_id)
            critic_positions = selected.positions
            critic_abs = selected.abs_witness
            critic_witness = selected.witness
            c_witness = float(np.mean(critic_abs)) if len(critic_abs) else np.nan
            c_fraction = float(len(critic_positions) / max(len(X), 1))
            c_distance = float(pairwise_distances(X[critic_positions], prev_X).min(axis=1).mean()) if len(critic_positions) else np.nan

            if len(curr_X) and len(critic_positions):
                old_labels = _nearest_prototype_labels(X[critic_positions], prev_X, prev_y)
                new_labels = _nearest_prototype_labels(X[critic_positions], curr_X, curr_y)
                valid = np.isfinite(old_labels) & np.isfinite(new_labels)
                c_reassignment = float(np.mean(old_labels[valid] != new_labels[valid])) if np.any(valid) else np.nan

            # Matched random-control score from the same candidate pool.
            rng = np.random.default_rng(seed * 100_003 + chunk_id)
            pool = selected.candidate_positions
            if len(pool):
                random_local = rng.choice(len(pool), min(n_criticisms, len(pool)), replace=False)
                random_positions = pool[random_local]
                random_witness = float(np.mean(selected.candidate_abs_witness[random_local]))
                random_region_fraction = float(block.iloc[random_positions]["gt_affected_region"].mean())
                err = np.asarray(errors, dtype=float)
                misclassified = pool[np.isfinite(err[pool]) & (err[pool] > 0.5)]
                n_error_criticism = int(len(misclassified))
                if len(misclassified):
                    # Error criticism uses misclassified candidates only, with variable K.
                    pool_lookup = {int(pos): j for j, pos in enumerate(pool)}
                    local_ids = [pool_lookup[int(pos)] for pos in misclassified]
                    error_witness = float(np.mean(selected.candidate_abs_witness[local_ids]))
                    error_region_fraction = float(block.iloc[misclassified]["gt_affected_region"].mean())

            curr_proto_region_fraction = (
                float(np.mean(block.iloc[curr_positions]["gt_affected_region"].to_numpy(dtype=bool)))
                if len(curr_positions) else np.nan
            )
            criticism_ms = (time.perf_counter() - criticism_started) * 1000.0

            for j, pos in enumerate(critic_positions):
                row = block.iloc[int(pos)]
                detail_rows.append({
                    "seed": seed,
                    "chunk": chunk_id,
                    "t": int(row["t"]),
                    "critic_position": int(pos),
                    "x1": float(row["x1"]),
                    "x2": float(row["x2"]),
                    "y": int(row["y"]),
                    "witness": float(critic_witness[j]),
                    "abs_witness": float(critic_abs[j]),
                    "nearest_previous_prototype_distance": float(pairwise_distances(X[[pos]], prev_X).min()),
                    "gt_affected_region": bool(row["gt_affected_region"]),
                    "gt_concept_drift": bool(row["gt_concept_drift"]),
                    "gt_event_class": str(row["gt_event_class"]),
                })

        elapsed_ms = (time.perf_counter() - total_started) * 1000.0
        chunk_gt = block["gt_event_class"].mode().iloc[0]
        raw_rows.append({
            "seed": seed,
            "chunk": chunk_id,
            "t_start": start,
            "t_end": end - 1,
            "stream_scope": scenario.scope,
            "drift_kind": scenario.drift_kind,
            "schedule": scenario.schedule,
            "extent": scenario.extent,
            "magnitude": scenario.magnitude,
            "chunk_size": chunk_size,
            "n_prototypes_previous": int(len(prev_X)),
            "n_prototypes_current": int(len(curr_X)),
            "p_pri": p_signal,
            "c_mean_abs_witness": c_witness,
            "c_mean_nearest_prototype_distance": c_distance,
            "c_criticism_reassignment_impact": c_reassignment,
            "c_criticism_fraction": c_fraction,
            "c_random_mean_abs_witness": random_witness,
            "c_random_in_affected_region_fraction": random_region_fraction,
            "c_error_mean_abs_witness": error_witness,
            "c_error_criticism_fraction": float(n_error_criticism / max(len(X), 1)),
            "n_error_criticism": n_error_criticism,
            "c_error_in_affected_region_fraction": error_region_fraction,
            "prequential_error_rate": float(np.nanmean(errors)) if np.any(np.isfinite(errors)) else np.nan,
            "racep_native_alarm_raw": bool(racep_native_chunk_flags[chunk_id]),
            "adwin_alarm_raw": bool(adwin_chunk_flags[chunk_id]),
            "adwin_detection_count_in_chunk": int(adwin_chunk_count[chunk_id]),
            "page_hinkley_alarm_raw": bool(page_hinkley_chunk_flags[chunk_id]),
            "page_hinkley_detection_count_in_chunk": int(page_hinkley_chunk_count[chunk_id]),
            "external_detector_update_ms": external_detector_ms,
            "n_criticisms": int(len(critic_positions)),
            "criticisms_in_affected_region_fraction": (
                float(block.iloc[critic_positions]["gt_affected_region"].mean()) if len(critic_positions) else np.nan
            ),
            "current_prototypes_in_affected_region_fraction": curr_proto_region_fraction,
            "ground_truth_class": str(chunk_gt),
            "gt_concept_drift": bool(block["gt_concept_drift"].any()),
            "gt_distribution_shift": bool(block["gt_distribution_shift"].any()),
            "model_update_ms": model_update_ms,
            "racep_model_update_ms": racep_model_update_ms,
            "racep_update_ms": racep_ms,
            "criticism_update_ms": criticism_ms,
            "total_chunk_processing_ms": elapsed_ms,
            "gamma_fixed": gamma,
        })
        previous_block_X = X.copy()

    raw = pd.DataFrame(raw_rows)
    detail = pd.DataFrame(detail_rows)
    # Match the official RACE-P warm-up convention: its reference metrics are
    # accumulated for chunk indices [grace_period/2, grace_period). C uses its
    # own stats on the same stationary calibration chunks.
    calibration_mask = (raw["chunk"] >= ref_chunks) & (raw["chunk"] < 2 * ref_chunks)
    if int(calibration_mask.sum()) < 2:
        raise ValueError("Need at least two stationary calibration chunks; increase n_chunks or reduce reference_chunks.")

    c_columns = [
        "c_mean_abs_witness",
        "c_mean_nearest_prototype_distance",
        "c_criticism_reassignment_impact",
        "c_random_mean_abs_witness",
        "c_error_mean_abs_witness",
        "c_error_criticism_fraction",
    ]
    detection_rows, zone_rows = [], []
    p_k3_strong = None
    true_starts = [scenario.drift_start // chunk_size] if scenario.drift_kind == "concept" else []
    gt_classes = raw["ground_truth_class"].astype(str).to_numpy()

    # Suppress decisions during the same warm-up/calibration period used by P/C,
    # while letting both external detectors observe every valid prequential error.
    for baseline_name, raw_flags, parameters in (
        ("RACE-P-original", racep_native_chunk_flags, {"const": 3.0, "grace_period": 2 * ref_chunks}),
        ("ADWIN", adwin_chunk_flags, {"delta": adwin_delta}),
        ("Page-Hinkley", page_hinkley_chunk_flags, {
            "delta": ph_delta, "threshold": ph_threshold, "alpha": ph_alpha,
            "min_instances": ph_min_instances, "mode": "up",
        }),
    ):
        eval_flags = raw_flags.copy()
        eval_flags[: 2 * ref_chunks] = False
        ev = event_detection_metrics(eval_flags, true_starts, len(raw), tolerance=event_tolerance)
        detection_rows.append({
            "seed": seed,
            "scope": scenario.scope,
            "drift_kind": scenario.drift_kind,
            "schedule": scenario.schedule,
            "extent": scenario.extent,
            "magnitude": scenario.magnitude,
            "chunk_size": chunk_size,
            "k": np.nan,
            "w": np.nan,
            "c_signal": "prequential_error",
            "rule": baseline_name,
            "baseline_parameters": json.dumps(parameters, sort_keys=True),
            "p_threshold": np.nan,
            "c_threshold": np.nan,
            "p_reference_mean": np.nan,
            "p_reference_std": np.nan,
            "c_reference_mean": np.nan,
            "c_reference_std": np.nan,
            "four_zone_chunk_accuracy": np.nan,
            "four_zone_balanced_accuracy_supported_classes": np.nan,
            **ev,
        })

    for k in k_values:
        p_cal = fit_upper_threshold(raw["p_pri"], calibration_mask, k)
        p_strong = above_threshold(raw["p_pri"], p_cal["threshold"])
        p_strong[: 2 * ref_chunks] = False  # warm-up/calibration chunks are not evaluated
        if np.isclose(k, 3.0):
            p_k3_strong = p_strong.copy()
        for c_col in c_columns:
            c_cal = fit_upper_threshold(raw[c_col], calibration_mask, k)
            c_strong = above_threshold(raw[c_col], c_cal["threshold"])
            c_strong[: 2 * ref_chunks] = False  # no decisions during warm-up/calibration
            zones = four_zone(p_strong, c_strong)
            zone_eval = four_zone_metrics(gt_classes, zones)
            for rule in ("P", "C", "OR", "AND"):
                for w in (w_values if rule == "AND" else [0]):
                    flags = p_strong if rule == "P" else c_strong if rule == "C" else combine_p_c(p_strong, c_strong, rule, w=w)
                    ev = event_detection_metrics(flags, true_starts, len(raw), tolerance=event_tolerance)
                    detection_rows.append({
                        "seed": seed,
                        "scope": scenario.scope,
                        "drift_kind": scenario.drift_kind,
                        "schedule": scenario.schedule,
                        "extent": scenario.extent,
                        "magnitude": scenario.magnitude,
                        "chunk_size": chunk_size,
                        "k": k,
                        "w": w,
                        "c_signal": c_col,
                        "rule": rule,
                        "p_threshold": p_cal["threshold"],
                        "c_threshold": c_cal["threshold"],
                        "p_reference_mean": p_cal["mean"],
                        "p_reference_std": p_cal["std"],
                        "c_reference_mean": c_cal["mean"],
                        "c_reference_std": c_cal["std"],
                        "four_zone_chunk_accuracy": zone_eval["accuracy"],
                        "four_zone_balanced_accuracy_supported_classes": zone_eval["balanced_accuracy_supported_classes"],
                        **ev,
                    })
            for i, z in enumerate(zones):
                zone_rows.append({
                    "seed": seed,
                    "chunk": int(i),
                    "k": k,
                    "c_signal": c_col,
                    "p_strong": bool(p_strong[i]),
                    "c_strong": bool(c_strong[i]),
                    "zone_prediction": str(z),
                    "ground_truth_class": str(gt_classes[i]),
                })

    native_scored = racep_native_chunk_flags.copy()
    native_scored[: 2 * ref_chunks] = False
    raw["racep_native_alarm_scored"] = native_scored
    if p_k3_strong is not None:
        raw["p_external_k3_alarm_scored"] = p_k3_strong
        evaluated = np.arange(len(raw)) >= 2 * ref_chunks
        agreement = float(np.mean(native_scored[evaluated] == p_k3_strong[evaluated])) if np.any(evaluated) else np.nan
        raw["native_vs_p_k3_alarm_agreement"] = agreement
        print(f"Native RACE-P vs external P(k=3) alarm agreement: {agreement:.3f}", flush=True)
    else:
        raw["p_external_k3_alarm_scored"] = np.nan
        raw["native_vs_p_k3_alarm_agreement"] = np.nan
        print("k=3 not in --k-values; skipped native RACE-P versus external P(k=3) agreement check.", flush=True)
    return raw, detail, pd.DataFrame(detection_rows), pd.DataFrame(zone_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "pilot", "final"], default="smoke")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--n-chunks", type=int, default=None)
    parser.add_argument("--reference-chunks", type=int, default=None)
    parser.add_argument("--max-streams", type=int, default=None)
    parser.add_argument("--scenario-offset", type=int, default=0, help="Starting row in the generated scenario grid; use to split final jobs into batches")
    parser.add_argument("--seed-start", type=int, default=None, help="Override profile seed start; default 1001 for pilot and 2001 for final")
    parser.add_argument("--n-seeds", type=int, default=None, help="Override seeds per scenario; default 3 for pilot and 20 for final")
    parser.add_argument("--n-criticisms", type=int, default=5)
    parser.add_argument("--k-values", default="1,2,3,4")
    parser.add_argument("--w-values", default="0,1,3,5")
    parser.add_argument("--event-tolerance", type=int, default=3, help="Common post-drift matching window Δ in chunks")
    parser.add_argument("--adwin-delta", type=float, default=0.002)
    parser.add_argument("--ph-delta", type=float, default=0.005)
    parser.add_argument("--ph-threshold", type=float, default=50.0)
    parser.add_argument("--ph-alpha", type=float, default=0.9999)
    parser.add_argument("--ph-min-instances", type=int, default=30)
    parser.add_argument("--output", default="results/thesis_rebuild_e2_e4")
    parser.add_argument("--run-id", default="run001")
    args = parser.parse_args()

    if args.chunk_size < 10:
        parser.error("chunk-size should be >= 10 for a meaningful pilot")
    seed_start = args.seed_start if args.seed_start is not None else (2001 if args.profile == "final" else 1001)
    n_seeds = args.n_seeds if args.n_seeds is not None else (20 if args.profile == "final" else 3)
    if args.profile == "smoke":
        n_seeds = 1
    n_chunks, ref_chunks, cases = _build_scenarios(
        args.profile, args.chunk_size, args.n_chunks, args.reference_chunks, seed_start, n_seeds
    )
    scenario_grid_total = len(cases)
    if args.scenario_offset < 0 or args.scenario_offset >= len(cases):
        parser.error(f"scenario-offset must be in [0, {max(len(cases)-1, 0)}]")
    cases = cases[args.scenario_offset:]
    if args.max_streams is not None:
        cases = cases[: args.max_streams]
    k_values = [float(x) for x in args.k_values.split(",")]
    w_values = [int(x) for x in args.w_values.split(",")]
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    output_names = [
        f"manifest_{args.run_id}.json",
        f"raw_signals_{args.run_id}.csv",
        f"criticism_details_{args.run_id}.csv",
        f"detection_metrics_{args.run_id}.csv",
        f"four_zone_chunks_{args.run_id}.csv",
    ]
    existing = [name for name in output_names if (outdir / name).exists()]
    if existing:
        parser.error(
            f"run-id {args.run_id!r} already has output files: {existing}. "
            "Choose a new --run-id so results are not overwritten."
        )
    manifest = {
        "run_id": args.run_id,
        "profile": args.profile,
        "seed_start": seed_start,
        "n_seeds_per_scenario_grid": n_seeds,
        "scenario_offset": args.scenario_offset,
        "max_streams": args.max_streams,
        "scenario_grid_total": scenario_grid_total,
        "scenario_grid_batch_count": len(cases),
        "chunk_size": args.chunk_size,
        "n_chunks": n_chunks,
        "reference_chunks": ref_chunks,
        "calibration_chunk_range_half_open": [ref_chunks, 2 * ref_chunks],
        "n_criticisms": args.n_criticisms,
        "mmd_critic": {"kernel": "RBF", "gamma_rule": "scale on initial stationary chunk", "max_candidates": 512, "max_current_reference": 1024, "diversity_weight": 0.05},
        "k_values": k_values,
        "w_values": w_values,
        "matching_tolerance_chunks": args.event_tolerance,
        "racep_native": {"const": 3.0, "grace_period": 2 * ref_chunks},
        "adwin": {"delta": args.adwin_delta},
        "page_hinkley": {"delta": args.ph_delta, "threshold": args.ph_threshold, "alpha": args.ph_alpha, "min_instances": args.ph_min_instances, "mode": "up_when_supported"},
        "c_signals": ["c_mean_abs_witness", "c_mean_nearest_prototype_distance", "c_criticism_reassignment_impact", "c_criticism_fraction"],
        "software_versions": {"python": __import__("platform").python_version(), "river": _version("river"), "protree": _version("protree"), "numpy": _version("numpy"), "pandas": _version("pandas"), "scikit_learn": _version("scikit-learn")},
        "git_commit": _git_commit(),
        "note": "Validate API/output on profile=smoke before pilot/final; native RaceP alarm is also recorded at const=3.0.",
    }
    (outdir / f"manifest_{args.run_id}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for i, (scenario_template, seed) in enumerate(cases, start=1):
        drift_start = scenario_template.drift_start
        if drift_start >= n_chunks * args.chunk_size:
            raise ValueError("drift_start is beyond stream end; increase --n-chunks")
        if drift_start // args.chunk_size < 2 * ref_chunks:
            raise ValueError("Drift must start after both warm-up and calibration windows; increase drift_start/n_chunks or reduce reference_chunks")
        scenario = Scenario(
            scope=scenario_template.scope,
            drift_kind=scenario_template.drift_kind,
            schedule=scenario_template.schedule,
            extent=scenario_template.extent,
            magnitude=scenario_template.magnitude,
            drift_start=drift_start,
            transition_width=scenario_template.transition_width,
        )
        frame = generate_controlled_stream(n_chunks * args.chunk_size, seed, scenario)
        print(f"[{i}/{len(cases)}] seed={seed} scope={scenario.scope} type={scenario.drift_kind} schedule={scenario.schedule} extent={scenario.extent} magnitude={scenario.magnitude}", flush=True)
        raw, detail, det, zones = run_one_stream(
            frame,
            scenario,
            seed,
            args.chunk_size,
            ref_chunks,
            k_values,
            w_values,
            args.n_criticisms,
            args.event_tolerance,
            adwin_delta=args.adwin_delta,
            ph_delta=args.ph_delta,
            ph_threshold=args.ph_threshold,
            ph_alpha=args.ph_alpha,
            ph_min_instances=args.ph_min_instances,
        )
        scenario_id = (
            f"{scenario.scope}_{scenario.drift_kind}_{scenario.schedule}_"
            f"e{scenario.extent:g}_{scenario.magnitude}_cs{args.chunk_size}_seed{seed}"
        )
        output_tables = (
            (raw, "raw_signals"),
            (detail, "criticism_details"),
            (det, "detection_metrics"),
            (zones, "four_zone_chunks"),
        )
        for table, stem in output_tables:
            if table.empty:
                continue
            table["scenario_id"] = scenario_id
            table["run_id"] = args.run_id
            table_path = outdir / f"{stem}_{args.run_id}.csv"
            table.to_csv(
                table_path,
                mode="a",
                header=(not table_path.exists()) or table_path.stat().st_size == 0,
                index=False,
            )
        # Incremental append after each stream limits memory and preserves results
        # if a later scenario crashes. Run IDs are never silently overwritten.

    print(f"\nSaved outputs under: {outdir}")


if __name__ == "__main__":
    main()
