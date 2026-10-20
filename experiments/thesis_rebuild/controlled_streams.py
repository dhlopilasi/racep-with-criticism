"""Controlled 2-D stream generator with explicit concept-drift ground truth.

The generator is intentionally independent of River/protree so it can be tested
before the detector integration is run. X is bounded in [0, 1]^2, which makes
feature scales comparable across scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd


Scope = Literal["local", "global", "mixed"]
DriftKind = Literal["concept", "virtual", "stationary"]
Schedule = Literal["abrupt", "gradual"]

MAGNITUDE_SHIFT = {"small": 0.10, "medium": 0.25, "large": 0.40}
VIRTUAL_MASS = {"small": 0.20, "medium": 0.50, "large": 0.80}


@dataclass(frozen=True)
class Scenario:
    scope: Scope = "local"
    drift_kind: DriftKind = "concept"
    schedule: Schedule = "abrupt"
    extent: float = 0.10
    magnitude: str = "medium"
    drift_start: int = 8_000
    transition_width: int = 2_000

    def validate(self) -> "Scenario":
        if self.scope not in {"local", "global", "mixed"}:
            raise ValueError(f"Unsupported scope: {self.scope}")
        if self.drift_kind not in {"concept", "virtual", "stationary"}:
            raise ValueError(f"Unsupported drift_kind: {self.drift_kind}")
        if self.schedule not in {"abrupt", "gradual"}:
            raise ValueError(f"Unsupported schedule: {self.schedule}")
        if not 0.0 < self.extent <= 1.0:
            raise ValueError("extent must be in (0, 1].")
        if self.magnitude not in MAGNITUDE_SHIFT:
            raise ValueError(f"Unsupported magnitude: {self.magnitude}")
        if self.drift_start < 0 or self.transition_width < 1:
            raise ValueError("drift_start must be >= 0 and transition_width >= 1.")
        if self.scope == "global" and self.extent != 1.0:
            raise ValueError("Global scenarios must use extent=1.0.")
        if self.drift_kind == "stationary" and self.scope != "global":
            # Scope is not meaningful without a drift; normalize it in the metadata.
            return self
        return self


def _alpha_at(t: np.ndarray, scenario: Scenario) -> np.ndarray:
    if scenario.drift_kind == "stationary":
        return np.zeros_like(t, dtype=float)
    after = t >= scenario.drift_start
    if scenario.schedule == "abrupt":
        return after.astype(float)
    alpha = (t - scenario.drift_start + 1) / scenario.transition_width
    return np.where(after, np.clip(alpha, 0.0, 1.0), 0.0)


def _local_region_mask(X: np.ndarray, extent: float) -> np.ndarray:
    """Centered vertical band with area exactly `extent` on the unit square.

    The band spans the full y-axis and therefore intersects the baseline decision
    boundary at every extent, avoiding tiny regions where the concept change would
    have almost no label impact.
    """
    half_width = float(extent) / 2.0
    lower = 0.5 - half_width
    upper = 0.5 + half_width
    return (X[:, 0] >= lower) & (X[:, 0] <= upper)


def generate_controlled_stream(
    n_samples: int,
    seed: int,
    scenario: Scenario,
) -> pd.DataFrame:
    """Generate an annotated stream.

    The local affected region is a vertical band of area `extent` spanning the
    full y-axis, centered on x1=0.5.

    Baseline concept: y = 1[x1 + x2 > 1].
    Concept drift shifts the decision boundary. A local drift applies the new
    boundary only inside a known centered vertical band; global drift applies
    it everywhere; mixed drift applies a smaller global shift and a stronger
    local shift in the band.

    Virtual drift changes P(X) but leaves the label function P(Y|X) unchanged.
    For a local virtual drift, samples are increasingly concentrated in the
    known vertical band. For a global virtual drift, both coordinates become increasingly
    skewed toward one. A stationary stream has no drift at all.

    `gt_affected_region` identifies the spatial support, not whether an individual
    label happened to change. `gt_label_changed` records actual label changes.
    """
    scenario.validate()
    if n_samples <= scenario.drift_start and scenario.drift_kind != "stationary":
        raise ValueError("n_samples must extend beyond drift_start for a drift scenario.")

    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=int)
    alpha = _alpha_at(t, scenario)
    X = rng.uniform(0.0, 1.0, size=(n_samples, 2))

    # Virtual drift only alters P(X); the conditional label rule below stays fixed.
    if scenario.drift_kind == "virtual":
        mass = VIRTUAL_MASS[scenario.magnitude]
        use_shifted_x = rng.random(n_samples) < (alpha * mass)
        if scenario.scope == "global" or scenario.extent >= 1.0:
            X[use_shifted_x] = rng.beta(
                a=1.0 + 3.0 * mass,
                b=1.0,
                size=(int(use_shifted_x.sum()), 2),
            )
        else:
            half_width = float(scenario.extent) / 2.0
            lower = 0.5 - half_width
            upper = 0.5 + half_width
            count = int(use_shifted_x.sum())
            X[use_shifted_x, 0] = rng.uniform(lower, upper, size=count)
            X[use_shifted_x, 1] = rng.uniform(0.0, 1.0, size=count)

    baseline_y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    y = baseline_y.copy()

    # Ground-truth support is recomputed after possible covariate resampling.
    region_mask = _local_region_mask(X, scenario.extent)
    if scenario.drift_kind == "stationary":
        support_mask = np.zeros(n_samples, dtype=bool)
    elif scenario.scope == "global":
        support_mask = np.ones(n_samples, dtype=bool)
    else:
        support_mask = region_mask.copy()

    if scenario.drift_kind == "concept":
        delta = MAGNITUDE_SHIFT[scenario.magnitude]
        if scenario.scope == "global":
            changed_y = (X[:, 0] + X[:, 1] > 1.0 + delta).astype(int)
            apply_new_concept = rng.random(n_samples) < alpha
            y[apply_new_concept] = changed_y[apply_new_concept]
        elif scenario.scope == "local":
            changed_y = (X[:, 0] + X[:, 1] > 1.0 + delta).astype(int)
            apply_new_concept = (rng.random(n_samples) < alpha) & region_mask
            y[apply_new_concept] = changed_y[apply_new_concept]
        else:  # mixed: global and a stronger local shift, allowing the 4th ground-truth class.
            global_delta = 0.5 * delta
            global_y = (X[:, 0] + X[:, 1] > 1.0 + global_delta).astype(int)
            local_y = (X[:, 0] + X[:, 1] > 1.0 + delta).astype(int)
            apply_global = rng.random(n_samples) < alpha
            y[apply_global] = global_y[apply_global]
            apply_local = (rng.random(n_samples) < alpha) & region_mask
            y[apply_local] = local_y[apply_local]

    gt_active = alpha > 0.0
    gt_concept_active = gt_active & (scenario.drift_kind == "concept")
    gt_distribution_active = gt_active & (scenario.drift_kind in {"concept", "virtual"})
    gt_label_changed = y != baseline_y

    if scenario.drift_kind == "stationary" or scenario.drift_kind == "virtual":
        gt_event_class = np.full(n_samples, "none", dtype=object)
    elif scenario.scope == "local":
        gt_event_class = np.where(gt_concept_active, "local", "none").astype(object)
    elif scenario.scope == "global":
        gt_event_class = np.where(gt_concept_active, "global", "none").astype(object)
    else:
        gt_event_class = np.where(gt_concept_active, "drift", "none").astype(object)

    return pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "y": y,
        "t": t,
        "drift_alpha": alpha,
        "gt_drift_active": gt_active,
        "gt_concept_drift": gt_concept_active,
        "gt_distribution_shift": gt_distribution_active,
        "gt_affected_region": support_mask,
        "gt_label_changed": gt_label_changed,
        "gt_event_class": gt_event_class,
        "gt_scope": scenario.scope,
        "gt_drift_kind": scenario.drift_kind,
        "gt_schedule": scenario.schedule,
        "gt_extent": scenario.extent,
        "gt_magnitude": scenario.magnitude,
        "seed": seed,
    })


def scenario_dict(scenario: Scenario) -> dict:
    """JSON/CSV friendly scenario metadata."""
    return asdict(scenario)
