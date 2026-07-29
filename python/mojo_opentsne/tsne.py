"""t-SNE objectives and optimization with an openTSNE-compatible surface."""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy import sparse
from scipy import fft as scipy_fft

from . import initialization as initialization_scheme
from ._lib import addr, f64, lib
from .affinity import Affinities, PerplexityBasedNN

EPSILON = np.finfo(np.float64).eps


class OptimizationInterrupt(RuntimeError):
    def __init__(self, error, final_embedding):
        super().__init__("Optimization was interrupted by a callback.")
        self.error = error
        self.final_embedding = final_embedding


def _csr(P):
    matrix = sparse.csr_matrix(P, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError("`P` must contain finite, non-negative probabilities.")
    indptr = np.ascontiguousarray(matrix.indptr)
    indices = np.ascontiguousarray(matrix.indices)
    if indptr.dtype not in (np.int32, np.int64) or indices.dtype != indptr.dtype:
        indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        indices = np.ascontiguousarray(indices, dtype=np.int64)
    return matrix, indptr, indices, f64(matrix.data)


def _objective_inputs(embedding, P, reference_embedding):
    embedding = f64(embedding)
    if embedding.ndim != 2 or not all(embedding.shape):
        raise ValueError("`embedding` must be a non-empty two-dimensional array.")
    if not np.isfinite(embedding).all():
        raise ValueError("`embedding` must contain only finite values.")
    reference = embedding if reference_embedding is None else f64(reference_embedding)
    if reference.ndim != 2 or not all(reference.shape):
        raise ValueError(
            "`reference_embedding` must be a non-empty two-dimensional array."
        )
    if reference.shape[1] != embedding.shape[1]:
        raise ValueError("Embedding and reference dimensions must match.")
    if not np.isfinite(reference).all():
        raise ValueError("`reference_embedding` must contain only finite values.")
    if sparse.csr_matrix(P).shape != (embedding.shape[0], reference.shape[0]):
        raise ValueError("`P` shape must match embedding and reference sample counts.")
    return embedding, reference


def _positive(embedding, reference, P, gradient, dof):
    matrix, indptr, indices, values = _csr(P)
    if values.size == 0:
        return 0.0, 0.0
    function = (
        lib().mot_positive_gradient_i32
        if indices.dtype == np.int32
        else lib().mot_positive_gradient
    )
    divergence = function(
        addr(indptr),
        addr(indices),
        addr(values),
        addr(embedding),
        addr(reference),
        addr(gradient),
        embedding.shape[0],
        reference.shape[0],
        embedding.shape[1],
        float(dof),
    )
    sum_p = lib().mot_sum_buffer(addr(values), values.size)
    return float(divergence), float(sum_p)


def _negative_exact(embedding, reference, gradient, dof, same):
    return float(
        lib().mot_negative_exact(
            addr(embedding),
            addr(reference),
            addr(gradient),
            embedding.shape[0],
            reference.shape[0],
            embedding.shape[1],
            float(dof),
            int(same),
        )
    )


def _negative_bh(embedding, reference, gradient, dof, theta, same):
    n_ref = reference.shape[0]
    capacity = max(32, 8 * n_ref + 16)
    scratch = [np.empty(capacity, dtype=np.float64) for _ in range(7)]
    children = np.empty(4 * capacity, dtype=np.float64)
    stack = np.empty(capacity, dtype=np.float64)
    sum_q = lib().mot_negative_bh(
        addr(embedding),
        addr(reference),
        addr(gradient),
        embedding.shape[0],
        n_ref,
        float(dof),
        float(theta),
        int(same),
        capacity,
        *(addr(array) for array in scratch),
        addr(children),
        addr(stack),
    )
    return float(sum_q)


def _next_power_of_two(value):
    return 1 << max(1, int(value - 1).bit_length())


def _fft_workers(cells, n_jobs):
    if cells < 256 * 256:
        return 1
    return 8 if n_jobs == 1 else n_jobs


def _negative_fft(embedding, gradient, dof, fft_params, n_jobs):
    if embedding.shape[1] != 2:
        raise RuntimeError("Interpolation t-SNE is supported only in two dimensions.")
    interval = float(fft_params.get("ints_in_interval", 1))
    interpolation_points = int(fft_params.get("n_interpolation_points", 3))
    min_intervals = int(fft_params.get("min_num_intervals", 50))
    if interval <= 0 or not math.isfinite(interval):
        raise ValueError("`ints_in_interval` must be finite and positive.")
    if interpolation_points <= 0 or min_intervals <= 0:
        raise ValueError("FFT interval counts must be positive.")
    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    span = float(np.max(hi - lo))
    target_spacing = interval / interpolation_points
    intervals = max(
        min_intervals * interpolation_points,
        int(math.ceil(span / target_spacing)) + 3,
    )
    data_side = _next_power_of_two(intervals)
    side = 2 * data_side
    spacing = max(target_spacing, span / max(1, data_side - 3))
    origin = lo - spacing
    cells = side * side
    workers = _fft_workers(cells, n_jobs)
    density = np.empty((side, side), dtype=np.complex128)
    field = np.empty((side, side), dtype=np.complex128)
    library = lib()
    library.mot_deposit_grid(
        addr(embedding),
        addr(density),
        embedding.shape[0],
        side,
        float(origin[0]),
        float(origin[1]),
        spacing,
    )
    scipy_fft.fft2(density, overwrite_x=True, workers=workers)
    for component in (1, 2):
        library.mot_make_kernel(
            addr(field), side, spacing, float(dof), component
        )
        scipy_fft.fft2(field, overwrite_x=True, workers=workers)
        library.mot_multiply_spectrum(addr(density), addr(field), cells)
        scipy_fft.ifft2(field, overwrite_x=True, workers=workers)
        library.mot_interpolate_field(
            addr(embedding),
            addr(field),
            addr(gradient),
            embedding.shape[0],
            side,
            float(origin[0]),
            float(origin[1]),
            spacing,
            component,
            float(dof),
        )
    library.mot_make_kernel(addr(field), side, spacing, float(dof), 0)
    scipy_fft.fft2(field, overwrite_x=True, workers=workers)
    library.mot_multiply_spectrum(addr(density), addr(field), cells)
    scipy_fft.ifft2(field, overwrite_x=True, workers=workers)
    sum_q = library.mot_interpolate_field(
        addr(embedding),
        addr(field),
        addr(gradient),
        embedding.shape[0],
        side,
        float(origin[0]),
        float(origin[1]),
        spacing,
        0,
        float(dof),
    )
    gradient *= -1.0 / sum_q if sum_q > 0 else 0.0
    return float(sum_q)


def kl_divergence_bh(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    del n_jobs
    dof = float(dof)
    if not math.isfinite(dof):
        raise ValueError("`dof` must be finite.")
    dof = max(dof, 1e-8)
    embedding, reference = _objective_inputs(embedding, P, reference_embedding)
    gradient = np.zeros_like(embedding)
    theta = float(bh_params.get("theta", 0.5))
    if theta < 0 or not math.isfinite(theta):
        raise ValueError("`theta` must be finite and non-negative.")
    if embedding.shape[1] != 2 or theta == 0:
        sum_q = _negative_exact(
            embedding, reference, gradient, dof, reference_embedding is None
        )
    else:
        sum_q = _negative_bh(
            embedding,
            reference,
            gradient,
            dof,
            theta,
            reference_embedding is None,
        )
        if sum_q < 0:
            sum_q = _negative_exact(embedding, reference, gradient, dof, True)
    divergence, sum_p = _positive(embedding, reference, P, gradient, dof)
    error = divergence + sum_p * math.log(sum_q + EPSILON)
    return (error if should_eval_error else 0.0), gradient


def kl_divergence_fft(
    embedding,
    P,
    dof,
    fft_params,
    reference_embedding=None,
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    dof = float(dof)
    if not math.isfinite(dof):
        raise ValueError("`dof` must be finite.")
    dof = max(dof, 1e-8)
    embedding, reference = _objective_inputs(embedding, P, reference_embedding)
    gradient = np.zeros_like(embedding)
    if reference_embedding is None:
        sum_q = _negative_fft(embedding, gradient, dof, fft_params, n_jobs)
    else:
        sum_q = _negative_exact(embedding, reference, gradient, dof, False)
    divergence, sum_p = _positive(embedding, reference, P, gradient, dof)
    error = divergence + sum_p * math.log(sum_q + EPSILON)
    return (error if should_eval_error else 0.0), gradient


class gradient_descent:
    def __init__(self):
        self.update = None
        self.gains = None

    def copy(self):
        return copy.deepcopy(self)

    def __call__(
        self,
        embedding,
        P,
        n_iter,
        objective_function,
        learning_rate="auto",
        momentum=0.8,
        exaggeration=None,
        dof=1,
        min_gain=0.01,
        max_grad_norm=None,
        max_step_norm=5,
        theta=0.5,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        reference_embedding=None,
        n_jobs=1,
        use_callbacks=False,
        callbacks=None,
        callbacks_every_iters=50,
        verbose=False,
    ):
        del verbose
        if learning_rate == "auto":
            learning_rate = embedding.shape[0] / (exaggeration or 1)
        if self.update is None or self.update.shape != embedding.shape:
            self.update = np.zeros_like(embedding)
            self.gains = np.ones_like(embedding)
        work_P = P if exaggeration in (None, 1) else P * exaggeration
        params = dict(
            dof=dof,
            bh_params={"theta": theta},
            fft_params={
                "n_interpolation_points": n_interpolation_points,
                "min_num_intervals": min_num_intervals,
                "ints_in_interval": ints_in_interval,
            },
            reference_embedding=reference_embedding,
            n_jobs=n_jobs,
        )
        error = 0.0
        for iteration in range(n_iter):
            call_callback = (
                use_callbacks
                and callbacks is not None
                and (iteration + 1) % callbacks_every_iters == 0
            )
            error, gradient = objective_function(
                embedding,
                work_P,
                should_eval_error=call_callback,
                **params,
            )
            if max_grad_norm is not None:
                norms = np.linalg.norm(gradient, axis=1)
                mask = norms > max_grad_norm
                gradient[mask] *= (max_grad_norm / (norms[mask] + 1e-6))[:, None]
            flipped = np.sign(self.update) != np.sign(gradient)
            self.gains[flipped] += 0.2
            self.gains[~flipped] = self.gains[~flipped] * 0.8 + min_gain
            self.update = momentum * self.update - learning_rate * self.gains * gradient
            if max_step_norm is not None:
                norms = np.linalg.norm(self.update, axis=1)
                mask = norms > max_step_norm
                self.update[mask] *= (max_step_norm / norms[mask])[:, None]
            embedding += self.update
            if reference_embedding is None:
                embedding -= embedding.mean(axis=0)
            if call_callback:
                callback_list = callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]
                if any(bool(callback(iteration + 1, error, embedding)) for callback in callback_list):
                    raise OptimizationInterrupt(error, embedding)
        error, _ = objective_function(
            embedding, work_P, should_eval_error=True, **params
        )
        if exaggeration not in (None, 1):
            error = error / exaggeration - math.log(exaggeration)
        return error, embedding


class TSNEEmbedding(np.ndarray):
    def __new__(
        cls,
        embedding,
        affinities,
        dof=1,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        negative_gradient_method="auto",
        random_state=None,
        optimizer=None,
        **gradient_descent_params,
    ):
        obj = np.array(embedding, dtype=np.float64, order="C").view(cls)
        if obj.shape[0] != affinities.P.shape[0]:
            raise ValueError("Embedding and affinities have different sample counts.")
        obj.affinities = affinities
        obj.gradient_descent_params = dict(gradient_descent_params)
        obj.gradient_descent_params.update(
            dof=dof,
            n_interpolation_points=n_interpolation_points,
            min_num_intervals=min_num_intervals,
            ints_in_interval=ints_in_interval,
            negative_gradient_method=negative_gradient_method,
        )
        obj.random_state = random_state
        obj.optimizer = gradient_descent() if optimizer is None else optimizer
        obj.kl_divergence = None
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        for name in (
            "affinities",
            "gradient_descent_params",
            "random_state",
            "optimizer",
            "kl_divergence",
        ):
            setattr(self, name, getattr(obj, name, None))

    def optimize(
        self,
        n_iter,
        inplace=False,
        propagate_exception=False,
        **gradient_descent_params,
    ):
        if inplace:
            embedding = self
        else:
            embedding = TSNEEmbedding(
                np.copy(self),
                self.affinities,
                random_state=self.random_state,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )
        params = dict(self.gradient_descent_params)
        params.update(gradient_descent_params)
        method = params.pop("negative_gradient_method", "auto")
        if method == "auto":
            method = "bh" if len(embedding) < 10_000 else "fft"
        if method in {"bh", "BH", "barnes-hut"}:
            objective = kl_divergence_bh
        elif method in {"fft", "FFT", "interpolation"}:
            objective = kl_divergence_fft
        elif callable(method):
            objective = method
        else:
            raise ValueError(f"Unrecognized negative gradient method `{method}`.")
        callbacks = params.get("callbacks")
        params["use_callbacks"] = callbacks is not None
        params["objective_function"] = objective
        try:
            error, result = embedding.optimizer(
                embedding=embedding,
                P=embedding.affinities.P,
                n_iter=n_iter,
                **params,
            )
        except OptimizationInterrupt as exc:
            if propagate_exception:
                raise
            error, result = exc.error, exc.final_embedding
        result.kl_divergence = error
        return result


class TSNE:
    def __init__(
        self,
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        early_exaggeration_iter=250,
        early_exaggeration="auto",
        n_iter=500,
        exaggeration=None,
        dof=1,
        theta=0.5,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        initialization="pca",
        metric="euclidean",
        metric_params=None,
        initial_momentum=0.8,
        final_momentum=0.8,
        max_grad_norm=None,
        max_step_norm=5,
        n_jobs=1,
        neighbors="auto",
        knn_kwargs=None,
        negative_gradient_method="auto",
        callbacks=None,
        callbacks_every_iters=50,
        random_state=None,
        verbose=False,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.early_exaggeration_iter = early_exaggeration_iter
        self.early_exaggeration = (
            max(12, exaggeration) if early_exaggeration == "auto" and exaggeration is not None
            else 12 if early_exaggeration == "auto" else early_exaggeration
        )
        self.n_iter = n_iter
        self.exaggeration = exaggeration
        self.dof = dof
        self.theta = theta
        self.n_interpolation_points = n_interpolation_points
        self.min_num_intervals = min_num_intervals
        self.ints_in_interval = ints_in_interval
        self.initialization = initialization
        self.metric = metric
        self.metric_params = metric_params
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.max_grad_norm = max_grad_norm
        self.max_step_norm = max_step_norm
        self.n_jobs = n_jobs
        self.neighbors = neighbors
        self.knn_kwargs = knn_kwargs
        self.negative_gradient_method = negative_gradient_method
        self.callbacks = callbacks
        self.callbacks_every_iters = callbacks_every_iters
        self.random_state = random_state
        self.verbose = verbose

    def prepare_initial(self, X=None, affinities=None, initialization=None):
        if X is None and affinities is None:
            raise ValueError("At least one of `X` and `affinities` is required.")
        if affinities is None:
            affinities = PerplexityBasedNN(
                X,
                self.perplexity,
                method=self.neighbors,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=self.verbose,
                knn_kwargs=self.knn_kwargs,
            )
        if not isinstance(affinities, Affinities):
            if not hasattr(affinities, "P"):
                raise ValueError("`affinities` must expose a sparse `P` matrix.")
        scheme = self.initialization if initialization is None else initialization
        n = affinities.P.shape[0]
        if isinstance(scheme, np.ndarray):
            embedding = np.array(scheme, copy=True)
        elif scheme == "pca" and X is not None:
            embedding = initialization_scheme.pca(
                X, self.n_components, random_state=self.random_state
            )
        elif scheme == "spectral" or (scheme == "pca" and X is None):
            embedding = initialization_scheme.spectral(
                affinities.P, self.n_components, random_state=self.random_state
            )
        elif scheme == "random":
            embedding = initialization_scheme.random(
                n, self.n_components, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unrecognized initialization scheme `{scheme}`.")
        return TSNEEmbedding(
            embedding,
            affinities,
            dof=self.dof,
            negative_gradient_method=self.negative_gradient_method,
            learning_rate=self.learning_rate,
            momentum=self.final_momentum,
            theta=self.theta,
            n_interpolation_points=self.n_interpolation_points,
            min_num_intervals=self.min_num_intervals,
            ints_in_interval=self.ints_in_interval,
            max_grad_norm=self.max_grad_norm,
            max_step_norm=self.max_step_norm,
            n_jobs=self.n_jobs,
            callbacks=self.callbacks,
            callbacks_every_iters=self.callbacks_every_iters,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def fit(self, X=None, affinities=None, initialization=None):
        embedding = self.prepare_initial(X, affinities, initialization)
        embedding.optimize(
            self.early_exaggeration_iter,
            exaggeration=self.early_exaggeration,
            momentum=self.initial_momentum,
            inplace=True,
            propagate_exception=True,
        )
        embedding.optimize(
            self.n_iter,
            exaggeration=self.exaggeration,
            momentum=self.final_momentum,
            inplace=True,
            propagate_exception=True,
        )
        return embedding
