from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


class MMDCritic:
    """
    MMD-Critic criticism selection.

    The prototype set is provided externally as row indices into X.
    Criticisms are selected using the original MMD-Critic objective:

        L(C) = sum_{l in C} | witness(x_l) |

    plus the diversity regularizer:

        r(K, C) = log det(K_CC)

    Criticisms are selected greedily from candidates outside the
    prototype set.
    """

    def __init__(
        self,
        n_criticisms: int = 3,
        gamma: float | str = "scale",
    ):
        self.n_criticisms = int(n_criticisms)
        self.gamma = gamma

    def _kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the global RBF kernel matrix."""
        if self.gamma == "scale":
            variance = np.var(X)

            if variance <= 0:
                gamma = 1.0
            else:
                gamma = 1.0 / (X.shape[1] * variance)

        else:
            gamma = float(self.gamma)

        return rbf_kernel(X, X, gamma=gamma)

    def _witness_scores(
        self,
        K: np.ndarray,
        prototype_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Original empirical MMD witness function:

            f(x_l) =
                mean_i k(x_l, x_i)
                -
                mean_j k(x_l, x_j), j in prototypes
        """
        data_mean = K.mean(axis=1)

        prototype_mean = K[:, prototype_indices].mean(axis=1)

        witness = data_mean - prototype_mean

        return witness

    def _objective_gain(
        self,
        candidate: int,
        selected: list[int],
        scores: np.ndarray,
        K: np.ndarray,
        jitter: float = 1e-10,
    ) -> float:
        """
        Marginal gain of:

            sum |witness| + log det(K_CC)

        when adding `candidate`.
        """
        linear_gain = float(scores[candidate])

        if not selected:
            # K[c,c] = 1 for RBF kernel, so log(det(K_cc)) = 0.
            return linear_gain

        current = np.asarray(selected, dtype=int)

        old_K = K[np.ix_(current, current)]

        new_indices = np.asarray(selected + [candidate], dtype=int)
        new_K = K[np.ix_(new_indices, new_indices)]

        sign_old, logdet_old = np.linalg.slogdet(
            old_K + jitter * np.eye(len(old_K))
        )

        sign_new, logdet_new = np.linalg.slogdet(
            new_K + jitter * np.eye(len(new_K))
        )

        if sign_new <= 0:
            return -np.inf

        if sign_old <= 0:
            logdet_gain = logdet_new
        else:
            logdet_gain = logdet_new - logdet_old

        return linear_gain + float(logdet_gain)

    def select(
        self,
        X,
        prototype_indices,
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        X:
            Full dataset used by RACE-P for the current explanation.
        prototype_indices:
            Indices of RACE-P prototypes in X.

        Returns
        -------
        criticism_indices:
            Selected criticism indices.
        criticism_scores:
            Absolute witness scores.
        witness:
            Signed witness values.
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.to_numpy(dtype=float)

        else:
            X_array = np.asarray(X, dtype=float)

        prototype_indices = np.asarray(
            list(prototype_indices),
            dtype=int,
        )

        n_samples = len(X_array)

        if n_samples == 0:
            return [], np.array([]), np.array([])

        if len(prototype_indices) == 0:
            raise ValueError(
                "MMD-Critic requires at least one prototype."
            )

        if np.any(prototype_indices < 0) or np.any(
            prototype_indices >= n_samples
        ):
            raise IndexError(
                "prototype_indices contain invalid indices."
            )

        K = self._kernel_matrix(X_array)

        witness = self._witness_scores(
            K,
            prototype_indices,
        )

        scores = np.abs(witness)

        prototype_mask = np.zeros(
            n_samples,
            dtype=bool,
        )

        prototype_mask[prototype_indices] = True

        candidates = [
            i
            for i in range(n_samples)
            if not prototype_mask[i]
        ]

        selected: list[int] = []

        for _ in range(
            min(self.n_criticisms, len(candidates))
        ):
            best_candidate = None
            best_gain = -np.inf

            for candidate in candidates:
                if candidate in selected:
                    continue

                gain = self._objective_gain(
                    candidate=candidate,
                    selected=selected,
                    scores=scores,
                    K=K,
                )

                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate

            if best_candidate is None:
                break

            selected.append(best_candidate)

        selected_array = np.asarray(
            selected,
            dtype=int,
        )

        return (
            selected,
            scores[selected_array],
            witness[selected_array],
        )
