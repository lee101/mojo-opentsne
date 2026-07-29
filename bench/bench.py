"""Measured mojo-opentsne versus openTSNE timings."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from scipy import sparse

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_opentsne as mot  # noqa: E402
from mojo_opentsne import affinity as mot_affinity  # noqa: E402
from openTSNE import affinity as ref_affinity  # noqa: E402
from openTSNE import tsne as ref_tsne  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def probabilities(n, k, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), k)
    cols = rng.integers(0, n, n * k)
    values = rng.uniform(0.1, 1, n * k)
    P = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    P.setdiag(0)
    P = P + P.T
    P /= P.sum()
    return P


def machine():
    cpu = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"{cpu}; {platform.system()} {platform.machine()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(0)
    cases = []

    X = np.ascontiguousarray(rng.normal(size=(5_000, 12)))
    cases.append(
        (
            "PerplexityBasedNN (5k x 12, p=30)",
            lambda: mot_affinity.PerplexityBasedNN(
                X, perplexity=30, method="exact", n_jobs=1
            ),
            lambda: ref_affinity.PerplexityBasedNN(
                X, perplexity=30, method="exact", n_jobs=1
            ),
        )
    )

    n = 8_000
    Y = np.ascontiguousarray(rng.normal(size=(n, 2)) * 8)
    P = probabilities(n, 30)
    cases.append(
        (
            "Barnes-Hut objective (8k, theta=.5)",
            lambda: mot.kl_divergence_bh(
                Y, P, 1, {"theta": 0.5}, should_eval_error=False
            ),
            lambda: ref_tsne.kl_divergence_bh(
                Y, P, 1, {"theta": 0.5}, should_eval_error=False, n_jobs=1
            ),
        )
    )

    n_exact = 2_000
    Y_exact = Y[:n_exact]
    P_exact = P[:n_exact, :n_exact].copy()
    P_exact /= P_exact.sum()
    cases.append(
        (
            "Exact objective (2k, theta=0)",
            lambda: mot.kl_divergence_bh(
                Y_exact, P_exact, 1, {"theta": 0}, should_eval_error=False
            ),
            lambda: ref_tsne.kl_divergence_bh(
                Y_exact,
                P_exact,
                1,
                {"theta": 0},
                should_eval_error=False,
                n_jobs=1,
            ),
        )
    )

    fft_params = {
        "n_interpolation_points": 3,
        "min_num_intervals": 50,
        "ints_in_interval": 1,
    }
    cases.append(
        (
            "FFT interpolation objective (8k)",
            lambda: mot.kl_divergence_fft(
                Y, P, 1, fft_params, should_eval_error=False
            ),
            lambda: ref_tsne.kl_divergence_fft(
                Y, P, 1, fft_params, should_eval_error=False, n_jobs=1
            ),
        )
    )

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-opentsne | openTSNE 1.0.4 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in cases:
        ours()
        reference()
        mojo_s = timeit(ours)
        ref_s = timeit(reference)
        ratio = ref_s / mojo_s
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {mojo_s * 1e3:.1f} ms | "
            f"{ref_s * 1e3:.1f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
