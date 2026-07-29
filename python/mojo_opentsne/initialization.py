"""Initialization schemes matching the covered openTSNE API."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state


def random(n_samples, n_components=2, random_state=None, verbose=False):
    del verbose
    rng = check_random_state(random_state)
    if isinstance(n_samples, np.ndarray):
        n_samples = n_samples.shape[0]
    return np.ascontiguousarray(rng.normal(0, 1e-4, (n_samples, n_components)))


def pca(
    X,
    n_components=2,
    svd_solver="auto",
    random_state=None,
    verbose=False,
    add_jitter=True,
):
    del verbose
    embedding = PCA(
        n_components=n_components,
        svd_solver=svd_solver,
        random_state=random_state,
    ).fit_transform(X)
    embedding /= np.std(embedding[:, 0]) * 10_000
    if add_jitter:
        rng = check_random_state(random_state)
        embedding += rng.normal(0, 1e-8, embedding.shape)
    return np.ascontiguousarray(embedding)


def spectral(
    A,
    n_components=2,
    tol=0.0001,
    max_iter=None,
    random_state=None,
    verbose=False,
    add_jitter=True,
):
    del verbose
    values, vectors = eigsh(
        A,
        k=n_components + 1,
        which="LM",
        tol=tol,
        maxiter=max_iter,
        v0=check_random_state(random_state).uniform(-1, 1, A.shape[0]),
    )
    embedding = vectors[:, np.argsort(values)[-2::-1][:n_components]]
    embedding /= np.std(embedding[:, 0]) * 10_000
    if add_jitter:
        embedding += check_random_state(random_state).normal(
            0, 1e-8, embedding.shape
        )
    return np.ascontiguousarray(embedding)
