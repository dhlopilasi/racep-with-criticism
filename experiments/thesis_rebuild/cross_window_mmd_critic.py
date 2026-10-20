"""Cross-window MMD-Critic adaptation for streaming experiments.

Candidate criticisms come from the CURRENT chunk, while reference prototypes
come from the PREVIOUS chunk. This avoids the old pilot's issue of fitting an
independent prototype set on each phase before comparing pre/post summaries.

This is a practical approximation: witness expectations are estimated from a
subsample of the current chunk and the greedy candidate pool is capped. It is
not claimed to be an exact reproduction of a static full-kernel MMD-Critic run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances


def resolve_rbf_gamma(X_reference: np.ndarray, gamma: str | float = "scale") -> float:
    X_reference = np.asarray(X_reference, dtype=float)
    if isinstance(gamma, (float, int)):
        if gamma <= 0:
            raise ValueError("Numeric gamma must be positive.")
        return float(gamma)
    if gamma == "scale":
        variance = float(np.var(X_reference))
        return 1.0 / max(X_reference.shape[1] * variance, 1e-12)
    if gamma == "median":
        sample = X_reference
        if len(sample) > 512:
            rng = np.random.default_rng(0)
            sample = sample[rng.choice(len(sample), 512, replace=False)]
        d2 = pairwise_distances(sample, metric="euclidean", squared=True)
        upper = d2[np.triu_indices_from(d2, k=1)]
        positive = upper[upper > 0]
        if not len(positive):
            return 1.0
        return 1.0 / float(np.median(positive))
    raise ValueError("gamma must be a positive number, 'scale', or 'median'.")


def _rbf_from_squared_distances(d2: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * np.maximum(d2, 0.0))


@dataclass
class CriticismResult:
    positions: np.ndarray
    abs_witness: np.ndarray
    witness: np.ndarray
    candidate_positions: np.ndarray
    candidate_abs_witness: np.ndarray
    candidate_witness: np.ndarray
    gamma: float


class CrossWindowMMDCritic:
    """Select current-chunk criticisms against a reference prototype set."""

    def __init__(
        self,
        n_criticisms: int = 5,
        gamma: float = 1.0,
        max_candidates: int = 512,
        max_current_reference: int = 1024,
        diversity_weight: float = 0.05,
        eps: float = 1e-10,
    ) -> None:
        if n_criticisms < 1:
            raise ValueError("n_criticisms must be >= 1")
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.n_criticisms = int(n_criticisms)
        self.gamma = float(gamma)
        self.max_candidates = int(max_candidates)
        self.max_current_reference = int(max_current_reference)
        self.diversity_weight = float(diversity_weight)
        self.eps = float(eps)

    def score_candidates(
        self,
        X_current: np.ndarray,
        X_reference_prototypes: np.ndarray,
        positions: np.ndarray | None = None,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return candidate positions and signed/absolute witness estimates."""
        X_current = np.asarray(X_current, dtype=float)
        X_reference_prototypes = np.asarray(X_reference_prototypes, dtype=float)
        if X_current.ndim != 2 or X_reference_prototypes.ndim != 2:
            raise ValueError("Current data and reference prototypes must be 2-D arrays")
        if X_current.shape[1] != X_reference_prototypes.shape[1]:
            raise ValueError("Current data and reference prototypes need same dimensions")
        if len(X_current) == 0 or len(X_reference_prototypes) == 0:
            raise ValueError("Current data and reference prototypes must be nonempty")

        rng = np.random.default_rng(seed)
        all_positions = np.arange(len(X_current), dtype=int)
        if positions is None:
            if len(all_positions) > self.max_candidates:
                positions = np.sort(rng.choice(all_positions, self.max_candidates, replace=False))
            else:
                positions = all_positions
        else:
            positions = np.asarray(positions, dtype=int)
            if np.any(positions < 0) or np.any(positions >= len(X_current)):
                raise ValueError("Candidate positions out of range")

        current_reference = X_current
        if len(current_reference) > self.max_current_reference:
            ref_idx = rng.choice(len(current_reference), self.max_current_reference, replace=False)
            current_reference = current_reference[ref_idx]

        candidates = X_current[positions]
        d2_current = pairwise_distances(candidates, current_reference, metric="euclidean", squared=True)
        d2_prototypes = pairwise_distances(candidates, X_reference_prototypes, metric="euclidean", squared=True)
        expected_current = _rbf_from_squared_distances(d2_current, self.gamma).mean(axis=1)
        expected_prototype = _rbf_from_squared_distances(d2_prototypes, self.gamma).mean(axis=1)
        witness = expected_current - expected_prototype
        return positions, np.abs(witness), witness

    def select_criticisms(
        self,
        X_current: np.ndarray,
        X_reference_prototypes: np.ndarray,
        seed: int = 0,
    ) -> CriticismResult:
        candidate_positions, candidate_abs, candidate_witness = self.score_candidates(
            X_current, X_reference_prototypes, seed=seed
        )
        if len(candidate_positions) == 0:
            raise ValueError("No candidates available")
        n_select = min(self.n_criticisms, len(candidate_positions))
        candidate_X = np.asarray(X_current, dtype=float)[candidate_positions]
        d2_candidate = pairwise_distances(
            candidate_X, candidate_X, metric="euclidean", squared=True
        )
        K = _rbf_from_squared_distances(d2_candidate, self.gamma)

        selected_local: list[int] = []
        remaining = np.ones(len(candidate_positions), dtype=bool)
        for _ in range(n_select):
            objective = candidate_abs.copy()
            if selected_local:
                K_ss = K[np.ix_(selected_local, selected_local)]
                K_ss = K_ss + self.eps * np.eye(len(selected_local))
                K_si = K[np.ix_(selected_local, np.arange(len(K)))]
                try:
                    solved = np.linalg.solve(K_ss, K_si)
                    conditional_variance = 1.0 - np.sum(K_si * solved, axis=0)
                except np.linalg.LinAlgError:
                    conditional_variance = np.ones(len(K))
                log_gain = np.log(np.maximum(conditional_variance, self.eps))
                objective += self.diversity_weight * log_gain
            objective[~remaining] = -np.inf
            best = int(np.argmax(objective))
            selected_local.append(best)
            remaining[best] = False

        selected_local_arr = np.asarray(selected_local, dtype=int)
        positions = candidate_positions[selected_local_arr]
        selected_abs = candidate_abs[selected_local_arr]
        selected_witness = candidate_witness[selected_local_arr]
        return CriticismResult(
            positions=positions,
            abs_witness=selected_abs,
            witness=selected_witness,
            candidate_positions=candidate_positions,
            candidate_abs_witness=candidate_abs,
            candidate_witness=candidate_witness,
            gamma=self.gamma,
        )
