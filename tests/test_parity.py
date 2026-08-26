"""Numerical and behavioral parity with openTSNE 1.0.4."""

from __future__ import annotations

import ctypes
import inspect

import numpy as np
import pytest
from scipy import sparse

import mojo_opentsne as mot
from mojo_opentsne import affinity, initialization
from mojo_opentsne._lib import addr, lib
from mojo_opentsne.tsne import _csr, _fft_workers, _positive

openTSNE = pytest.importorskip("openTSNE")
from openTSNE import affinity as upstream_affinity
from openTSNE import initialization as upstream_initialization
from openTSNE import tsne as upstream_tsne


def sparse_probabilities(n, k=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = rng.integers(0, n, n * k)
    values = rng.uniform(0.1, 1, n * k)
    P = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    P.setdiag(0)
    P = P + P.T
    P /= P.sum()
    return P


@pytest.fixture(scope="module")
def embedding():
    return np.ascontiguousarray(np.random.default_rng(4).normal(size=(300, 2)))


def test_shared_library_exports():
    library = lib()
    assert isinstance(library, ctypes.CDLL)
    for name in (
        "mot_positive_gradient",
        "mot_positive_gradient_i32",
        "mot_negative_exact",
        "mot_negative_bh",
        "mot_negative_fft",
        "mot_deposit_grid",
        "mot_make_kernel",
        "mot_multiply_spectrum",
        "mot_interpolate_field",
        "mot_scale_buffer",
        "mot_sum_buffer",
        "mot_perplexity_probabilities",
        "mot_multiscale_probabilities",
    ):
        assert getattr(library, name)


@pytest.mark.parametrize("theta", [0, 0.2, 0.5, 0.8])
def test_barnes_hut_objective_parity(embedding, theta):
    P = sparse_probabilities(len(embedding), seed=1)
    ours_error, ours_grad = mot.kl_divergence_bh(
        embedding, P, 1, {"theta": theta}, should_eval_error=True
    )
    ref_error, ref_grad = upstream_tsne.kl_divergence_bh(
        embedding,
        P,
        1,
        {"theta": theta},
        should_eval_error=True,
        n_jobs=1,
    )
    assert ours_error == pytest.approx(ref_error, abs=2e-9)
    assert np.allclose(ours_grad, ref_grad, atol=2e-14, rtol=2e-11)


@pytest.mark.parametrize("dof", [0.5, 1, 2, 5])
def test_exact_generalized_dof_parity(embedding, dof):
    P = sparse_probabilities(len(embedding), seed=2)
    ours = mot.kl_divergence_bh(
        embedding, P, dof, {"theta": 0}, should_eval_error=True
    )
    reference = upstream_tsne.kl_divergence_bh(
        embedding,
        P,
        dof,
        {"theta": 0},
        should_eval_error=True,
        n_jobs=1,
    )
    assert ours[0] == pytest.approx(reference[0], abs=2e-9)
    assert np.allclose(ours[1], reference[1], atol=1e-13, rtol=2e-11)


def test_reference_embedding_objective_parity():
    rng = np.random.default_rng(9)
    query = np.ascontiguousarray(rng.normal(size=(30, 2)))
    reference = np.ascontiguousarray(rng.normal(size=(80, 2)))
    values = rng.random((30, 80))
    values[values < 0.92] = 0
    P = sparse.csr_matrix(values / values.sum())
    ours = mot.kl_divergence_bh(
        query,
        P,
        1,
        {"theta": 0.5},
        reference_embedding=reference,
        should_eval_error=True,
    )
    theirs = upstream_tsne.kl_divergence_bh(
        query,
        P,
        1,
        {"theta": 0.5},
        reference_embedding=reference,
        should_eval_error=True,
        n_jobs=1,
    )
    assert ours[0] == pytest.approx(theirs[0], abs=2e-9)
    assert np.allclose(ours[1], theirs[1], atol=2e-14, rtol=2e-11)


@pytest.mark.parametrize("interval", [0.5, 0.25])
def test_fft_interpolation_parity(embedding, interval):
    P = sparse_probabilities(len(embedding), seed=3)
    params = {
        "n_interpolation_points": 3,
        "min_num_intervals": 50,
        "ints_in_interval": interval,
    }
    ours_error, ours_grad = mot.kl_divergence_fft(
        embedding, P, 1, params, should_eval_error=True
    )
    ref_error, ref_grad = upstream_tsne.kl_divergence_fft(
        embedding, P, 1, params, should_eval_error=True, n_jobs=1
    )
    cosine = np.vdot(ours_grad, ref_grad) / (
        np.linalg.norm(ours_grad) * np.linalg.norm(ref_grad)
    )
    assert ours_error == pytest.approx(ref_error, abs=0.035 if interval == 0.5 else 0.01)
    assert cosine > (0.998 if interval == 0.5 else 0.9998)
    assert np.max(np.abs(ours_grad - ref_grad)) < (
        2.5e-4 if interval == 0.5 else 8e-5
    )


def test_fft_rejects_more_than_two_dimensions():
    Y = np.zeros((10, 3))
    with pytest.raises(RuntimeError, match="two dimensions"):
        mot.kl_divergence_fft(
            Y,
            sparse_probabilities(10, k=3),
            1,
            {"min_num_intervals": 50, "ints_in_interval": 1},
        )


def test_fft_parallel_threshold():
    assert _fft_workers(256 * 256 - 1, 1) == 1
    assert _fft_workers(256 * 256, 1) == 8
    assert _fft_workers(256 * 256, 3) == 3


def test_simd_scalar_tail():
    values = np.arange(7, dtype=np.float64)
    assert lib().mot_sum_buffer(addr(values), len(values)) == 21
    lib().mot_scale_buffer(addr(values), len(values), 2.5)
    assert np.array_equal(values, np.arange(7, dtype=np.float64) * 2.5)


def test_positive_gradient_simd_tail_and_error_elision():
    rng = np.random.default_rng(31)
    embedding = np.ascontiguousarray(rng.normal(size=(9, 5)))
    P = sparse_probabilities(len(embedding), k=4, seed=32)
    expected_gradient = np.zeros_like(embedding)
    expected_divergence = 0.0
    for i in range(len(embedding)):
        for edge in range(P.indptr[i], P.indptr[i + 1]):
            j = P.indices[edge]
            probability = P.data[edge]
            delta = embedding[i] - embedding[j]
            dist2 = np.dot(delta, delta)
            inv = 1.0 / (1.0 + dist2)
            expected_gradient[i] += probability * inv * delta
            expected_divergence += probability * (
                np.log(probability) - np.log(inv)
            )

    evaluated_gradient = np.zeros_like(embedding)
    divergence, sum_p = _positive(
        embedding, embedding, P, evaluated_gradient, 1.0, True
    )
    elided_gradient = np.zeros_like(embedding)
    elided_divergence, elided_sum_p = _positive(
        embedding, embedding, P, elided_gradient, 1.0, False
    )

    assert divergence == pytest.approx(expected_divergence, abs=3e-11)
    assert sum_p == pytest.approx(P.sum(), abs=2e-16)
    assert np.allclose(evaluated_gradient, expected_gradient, atol=2e-17, rtol=2e-15)
    assert np.array_equal(elided_gradient, evaluated_gradient)
    assert elided_divergence == 0
    assert elided_sum_p == 0


def test_csr_indices_cross_ffi_without_copy():
    P = sparse_probabilities(20, k=3)
    matrix, indptr, indices, values = _csr(P)
    assert np.shares_memory(indptr, matrix.indptr)
    assert np.shares_memory(indices, matrix.indices)
    assert np.shares_memory(values, matrix.data)


def test_ffi_buffer_validation_and_objective_shapes():
    with pytest.raises(ValueError, match="empty"):
        addr(np.empty(0, dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        addr(np.ones((3, 3), dtype=np.float64)[:, ::2])
    with pytest.raises(ValueError, match="non-empty"):
        mot.kl_divergence_bh(
            np.empty((0, 2)), sparse.csr_matrix((0, 0)), 1, {"theta": 0.5}
        )
    with pytest.raises(ValueError, match="shape"):
        mot.kl_divergence_bh(
            np.ones((3, 2)), sparse.eye(2), 1, {"theta": 0.5}
        )
    with pytest.raises(ValueError, match="non-negative"):
        mot.kl_divergence_bh(
            np.ones((3, 2)), -sparse.eye(3), 1, {"theta": 0.5}
        )
    with pytest.raises(ValueError, match="finite"):
        mot.kl_divergence_bh(
            np.ones((3, 2)), sparse.eye(3), np.nan, {"theta": 0.5}
        )


def test_invalid_fft_parameters_are_rejected():
    Y = np.ones((3, 2))
    with pytest.raises(ValueError, match="positive"):
        mot.kl_divergence_fft(
            Y,
            sparse.eye(3),
            1,
            {"ints_in_interval": 0, "min_num_intervals": 50},
        )


def test_perplexity_affinity_parity():
    X = np.random.default_rng(10).normal(size=(180, 7))
    ours = affinity.PerplexityBasedNN(
        X, perplexity=20, method="exact", n_jobs=1
    )
    theirs = upstream_affinity.PerplexityBasedNN(
        X, perplexity=20, method="exact", n_jobs=1
    )
    assert ours.P.sum() == pytest.approx(1)
    assert ours.P.nnz == theirs.P.nnz
    assert np.max(np.abs((ours.P - theirs.P).data)) < 5e-10


def test_set_perplexity_parity():
    X = np.random.default_rng(11).normal(size=(120, 4))
    ours = affinity.PerplexityBasedNN(X, perplexity=20, method="exact")
    theirs = upstream_affinity.PerplexityBasedNN(X, perplexity=20, method="exact")
    ours.set_perplexity(10)
    theirs.set_perplexity(10)
    assert np.max(np.abs((ours.P - theirs.P).data)) < 5e-10


def test_multiscale_mixture_parity():
    X = np.random.default_rng(18).normal(size=(120, 4))
    ours = affinity.MultiscaleMixture(
        X, perplexities=(5, 15), method="exact", n_jobs=1
    )
    theirs = upstream_affinity.MultiscaleMixture(
        X, perplexities=(5, 15), method="exact", n_jobs=1
    )
    assert np.max(np.abs((ours.P - theirs.P).data)) < 5e-10


def test_random_initialization_parity():
    ours = initialization.random(50, 2, random_state=42)
    theirs = upstream_initialization.random(50, 2, random_state=42)
    assert np.array_equal(ours, theirs)
    assert ours.flags.c_contiguous


def test_pca_initialization_parity():
    X = np.random.default_rng(12).normal(size=(100, 5))
    ours = initialization.pca(X, 2, random_state=42, add_jitter=False)
    theirs = upstream_initialization.pca(X, 2, random_state=42, add_jitter=False)
    assert np.allclose(ours, theirs, atol=1e-15)


def test_spectral_initialization_is_finite_and_scaled():
    P = sparse_probabilities(80, seed=19)
    result = initialization.spectral(P, 2, random_state=42, add_jitter=False)
    assert result.shape == (80, 2)
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()
    assert np.std(result[:, 0]) == pytest.approx(1e-4)


def test_embedding_optimization_parity():
    X = np.random.default_rng(13).normal(size=(160, 5))
    ours_aff = affinity.PerplexityBasedNN(X, perplexity=15, method="exact")
    ref_aff = upstream_affinity.PerplexityBasedNN(X, perplexity=15, method="exact")
    init = initialization.random(len(X), random_state=7)
    ours = mot.TSNEEmbedding(
        init,
        ours_aff,
        learning_rate=10,
        momentum=0.5,
        negative_gradient_method="bh",
        theta=0.5,
    ).optimize(15)
    theirs = openTSNE.TSNEEmbedding(
        init,
        ref_aff,
        learning_rate=10,
        momentum=0.5,
        negative_gradient_method="bh",
        theta=0.5,
    ).optimize(15)
    # Upstream can drop points that lie within 1e-6 of an internal tree
    # center; this tree retains them, so trajectories differ slightly at the
    # deliberately tiny 1e-4 initialization scale.
    assert np.allclose(ours, theirs, atol=3e-6, rtol=2e-9)
    assert ours.kl_divergence == pytest.approx(theirs.kl_divergence, abs=2e-9)


def test_optimize_copy_and_inplace_behavior():
    X = np.random.default_rng(14).normal(size=(80, 3))
    affinities = affinity.PerplexityBasedNN(X, perplexity=10, method="exact")
    original = mot.TSNEEmbedding(
        initialization.random(80, random_state=1), affinities, learning_rate=5
    )
    before = original.copy()
    copied = original.optimize(2, inplace=False)
    assert np.array_equal(original, before)
    assert not np.array_equal(copied, original)
    returned = original.optimize(2, inplace=True)
    assert returned is original
    assert not np.array_equal(original, before)


def test_callback_can_interrupt():
    X = np.random.default_rng(15).normal(size=(60, 3))
    affinities = affinity.PerplexityBasedNN(X, perplexity=8, method="exact")
    calls = []

    def callback(iteration, error, embedding):
        calls.append((iteration, error, embedding.shape))
        return True

    embedding = mot.TSNEEmbedding(
        initialization.random(60, random_state=2),
        affinities,
        learning_rate=5,
        callbacks=callback,
        callbacks_every_iters=2,
    )
    result = embedding.optimize(10)
    assert calls[0][0] == 2
    assert result.shape == (60, 2)


def test_tsne_signatures_match_covered_upstream_surface():
    assert inspect.signature(mot.TSNE) == inspect.signature(openTSNE.TSNE)
    assert inspect.signature(mot.TSNE.fit) == inspect.signature(openTSNE.TSNE.fit)
    assert inspect.signature(mot.TSNEEmbedding.optimize) == inspect.signature(
        openTSNE.TSNEEmbedding.optimize
    )
    assert inspect.signature(affinity.PerplexityBasedNN) == inspect.signature(
        upstream_affinity.PerplexityBasedNN
    )


def test_end_to_end_tsne_separates_clusters():
    rng = np.random.default_rng(16)
    X = np.vstack(
        [rng.normal(-2, 0.4, (60, 4)), rng.normal(2, 0.4, (60, 4))]
    )
    result = mot.TSNE(
        perplexity=15,
        initialization="pca",
        early_exaggeration_iter=20,
        n_iter=30,
        learning_rate=20,
        random_state=1,
        neighbors="exact",
    ).fit(X)
    within = np.mean(
        np.linalg.norm(result[:60] - result[:60].mean(axis=0), axis=1)
    )
    between = np.linalg.norm(
        result[:60].mean(axis=0) - result[60:].mean(axis=0)
    )
    assert isinstance(result, mot.TSNEEmbedding)
    assert np.isfinite(result).all()
    assert np.isfinite(result.kl_divergence)
    assert between > 3 * within
