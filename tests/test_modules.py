import numpy as np
import pandas as pd

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


def test_controlled_local_drift_has_ground_truth():
    s = Scenario(scope="local", drift_kind="concept", schedule="abrupt", extent=0.1,
                 magnitude="large", drift_start=500, transition_width=100)
    df = generate_controlled_stream(1000, 7, s)
    assert df.shape[0] == 1000
    assert df.loc[df.t < 500, "gt_event_class"].eq("none").all()
    assert df.loc[df.t >= 500, "gt_event_class"].eq("local").all()
    assert df["gt_affected_region"].mean() < 0.2


def test_virtual_drift_does_not_change_label_function_definition():
    s = Scenario(scope="local", drift_kind="virtual", schedule="abrupt", extent=0.1,
                 magnitude="large", drift_start=500, transition_width=100)
    df = generate_controlled_stream(1000, 8, s)
    assert df.loc[df.t >= 500, "gt_event_class"].eq("none").all()
    assert not df.loc[df.t >= 500, "gt_concept_drift"].any()
    assert df.loc[df.t >= 500, "gt_distribution_shift"].all()


def test_cross_window_critic_returns_unique_current_positions():
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 0.1, size=(12, 3))
    cur = np.vstack([rng.normal(0, 0.1, size=(90, 3)), rng.normal(1, 0.1, size=(10, 3))])
    gamma = resolve_rbf_gamma(ref, "scale")
    critic = CrossWindowMMDCritic(n_criticisms=5, gamma=gamma, max_candidates=64)
    result = critic.select_criticisms(cur, ref, seed=11)
    assert len(result.positions) == 5
    assert len(set(result.positions.tolist())) == 5
    assert np.all(result.positions >= 0) and np.all(result.positions < len(cur))
    assert np.all(np.isfinite(result.witness))


def test_threshold_and_rules_are_causal():
    values = np.array([0.1, 0.2, 0.15, 0.3, 0.7])
    ref_mask = np.array([True, True, True, False, False])
    cal = fit_upper_threshold(values, ref_mask, 2)
    strong = above_threshold(values, cal["threshold"])
    assert strong.tolist() == [False, False, False, True, True]
    p = np.array([False, True, False, False])
    c = np.array([False, False, True, False])
    assert combine_p_c(p, c, "AND", w=1).tolist() == [False, False, True, False]
    assert four_zone(p, c).tolist() == ["none", "global", "local", "none"]


def test_event_matching_and_four_zone_confusion():
    flags = np.array([False, False, True, False, False, True])
    metrics = event_detection_metrics(flags, [2, 5], n_chunks=6, tolerance=1)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    zone = four_zone_metrics(["none", "global", "local", "drift"],
                             ["none", "global", "local", "drift"])
    assert zone["accuracy"] == 1.0


def test_e2_aggregation_keeps_seed_as_replicate_not_condition():
    from experiments.thesis_rebuild.analyze_e2_e7 import analyze_e2, aggregate_e2

    rows = []
    for seed in (1, 2, 3):
        for chunk in range(10):
            drift = chunk >= 5
            rows.append({
                "scenario_id": f"local_seed{seed}",
                "seed": seed,
                "stream_scope": "local",
                "drift_kind": "concept",
                "schedule": "abrupt",
                "extent": 0.1,
                "magnitude": "medium",
                "chunk_size": 100,
                "chunk": chunk,
                "gt_concept_drift": drift,
                "p_pri": float(chunk) + seed * 0.01,
            })
    per_seed = analyze_e2(pd.DataFrame(rows))
    aggregate = aggregate_e2(per_seed)
    row = aggregate[aggregate["signal"].eq("p_pri")].iloc[0]
    assert row["roc_auc_concept_drift_mean"] == 1.0
    assert row["roc_auc_concept_drift_count"] == 3


def test_e7_paired_tests_are_seed_level_and_holm_adjusted():
    from experiments.thesis_rebuild.analyze_e2_e7 import paired_e7_tests

    frame = pd.DataFrame({
        "seed": [1, 2, 3, 4, 5],
        "extent": [0.1] * 5,
        "magnitude": ["medium"] * 5,
        "schedule": ["abrupt"] * 5,
        "chunk_size": [100] * 5,
        "criticism_minus_prototype": [0.2, 0.1, 0.15, 0.08, 0.12],
        "criticism_minus_random": [0.1, 0.1, 0.05, 0.08, 0.12],
    })
    result = paired_e7_tests(frame)
    assert len(result) == 2
    assert result["n_seeds"].eq(5).all()
    assert result["mean_paired_difference"].gt(0).all()
    assert "wilcoxon_p_holm" in result.columns
