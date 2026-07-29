"""Sparse nearest-neighbor affinities for t-SNE."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ._lib import addr, f64, lib


class Affinities:
    def __init__(self, verbose=False):
        self.P = None
        self.verbose = verbose
        self.knn_index = None

    @property
    def n_samples(self):
        return self.P.shape[0]


def _joint_probabilities(neighbors, distances, perplexity, symmetrize=True):
    neighbors = np.asarray(neighbors)
    distances = np.asarray(distances)
    if neighbors.ndim != 2 or distances.shape != neighbors.shape:
        raise ValueError("Neighbor indices and distances must be matching 2D arrays.")
    n, k = neighbors.shape
    if n == 0 or k == 0:
        raise ValueError("Neighbor arrays must have at least one row and one neighbor.")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise TypeError("Neighbor indices must have an integer dtype.")
    if np.any(neighbors < 0) or np.any(neighbors >= n):
        raise ValueError("Neighbor indices are out of bounds.")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Neighbor distances must be finite and non-negative.")
    with np.errstate(over="ignore"):
        distances2 = f64(distances * distances)
    if not np.isfinite(distances2).all():
        raise ValueError("Squared neighbor distances exceed float64 range.")
    conditional = np.empty_like(distances2)
    perplexities = np.atleast_1d(perplexity).astype(np.float64)
    if perplexities.size == 1:
        lib().mot_perplexity_probabilities(
            addr(distances2), addr(conditional), n, k, float(perplexities[0])
        )
    else:
        perplexities = np.ascontiguousarray(perplexities)
        lib().mot_multiscale_probabilities(
            addr(distances2),
            addr(conditional),
            addr(perplexities),
            n,
            k,
            perplexities.size,
        )
    rows = np.repeat(np.arange(n), k)
    P = sparse.csr_matrix(
        (conditional.ravel(), (rows, neighbors.ravel())), shape=(n, n)
    )
    P /= n
    if symmetrize:
        P = (P + P.T) / 2
    P.sum_duplicates()
    P.sort_indices()
    return P


class PerplexityBasedNN(Affinities):
    def __init__(
        self,
        data=None,
        perplexity=30,
        method="auto",
        metric="euclidean",
        metric_params=None,
        symmetrize=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        k_neighbors="auto",
        knn_kwargs=None,
        knn_index=None,
    ):
        del random_state
        if data is None and knn_index is None:
            raise ValueError(
                "At least one of the parameters `data` or `knn_index` must be specified!"
            )
        if data is not None and knn_index is not None:
            raise ValueError("Pass only one of `data` and `knn_index`.")
        if knn_index is not None:
            neighbors, distances = knn_index.build()
            self.knn_index = knn_index
        else:
            data = np.asarray(data)
            if data.ndim != 2 or data.shape[0] < 2:
                raise ValueError("`data` must be a 2D array with at least two samples.")
            n = data.shape[0]
            k = min(n - 1, int(3 * perplexity)) if k_neighbors == "auto" else int(k_neighbors)
            effective = self.check_perplexity(perplexity, k)
            algorithm = "brute" if method == "exact" else "auto"
            kwargs = dict(knn_kwargs or {})
            index = NearestNeighbors(
                n_neighbors=k + 1,
                metric=metric,
                metric_params=metric_params,
                algorithm=algorithm,
                n_jobs=n_jobs,
                **kwargs,
            ).fit(data)
            distances, neighbors = index.kneighbors(data)
            distances, neighbors = distances[:, 1:], neighbors[:, 1:]
            self.knn_index = index
            perplexity = effective
        self._neighbors = np.ascontiguousarray(neighbors, dtype=np.int64)
        self._distances = f64(distances)
        self.P = _joint_probabilities(
            self._neighbors, self._distances, perplexity, symmetrize
        )
        self.perplexity = perplexity
        self.effective_perplexity_ = perplexity
        self.symmetrize = symmetrize
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.knn_kwargs = knn_kwargs

    @staticmethod
    def check_perplexity(perplexity, k_neighbors):
        if perplexity <= 0:
            raise ValueError("Perplexity must be positive.")
        if perplexity > k_neighbors:
            effective = k_neighbors / 3
            warnings.warn(
                f"Perplexity value {perplexity} is too high. Using perplexity {effective:.2f} instead",
                RuntimeWarning,
            )
            return effective
        return float(perplexity)

    def set_perplexity(self, new_perplexity):
        effective = self.check_perplexity(new_perplexity, self._neighbors.shape[1])
        k_neighbors = int(3 * new_perplexity)
        self.P = _joint_probabilities(
            self._neighbors[:, :k_neighbors],
            self._distances[:, :k_neighbors],
            effective,
            self.symmetrize,
        )
        self.perplexity = new_perplexity
        self.effective_perplexity_ = effective


class MultiscaleMixture(PerplexityBasedNN):
    def __init__(self, data=None, perplexities=(10, 100), **kwargs):
        values = np.atleast_1d(perplexities).astype(float)
        max_perplexity = float(values.max())
        super().__init__(data, perplexity=max_perplexity, **kwargs)
        effective = np.array(
            [
                self.check_perplexity(value, self._neighbors.shape[1])
                for value in values
            ]
        )
        self.P = _joint_probabilities(
            self._neighbors, self._distances, effective, self.symmetrize
        )
        self.perplexities = list(values)
        self.effective_perplexities_ = list(effective)
