# mojo-opentsne

`mojo-opentsne` is a standalone Mojo implementation of the compute-heavy
parts of [openTSNE](https://github.com/pavlin-policar/openTSNE), exposed to
Python through an API compatible with the covered openTSNE 1.0.4 subset. It
implements sparse attractive forces, exact and Barnes-Hut repulsive forces,
FFT-accelerated particle-mesh repulsion, and perplexity probability search.

```python
import numpy as np
import mojo_opentsne as openTSNE

rng = np.random.default_rng(0)
X = np.vstack([
    rng.normal(-2, 0.5, (100, 4)),
    rng.normal(2, 0.5, (100, 4)),
])

embedding = openTSNE.TSNE(
    perplexity=20,
    initialization="pca",
    negative_gradient_method="bh",
    early_exaggeration_iter=50,
    n_iter=100,
    random_state=0,
).fit(X)

print(embedding.shape, embedding.kl_divergence)
```

The different import name allows it to be installed beside the reference
package for parity testing. For the covered surface, changing
`import openTSNE` to `import mojo_opentsne as openTSNE` is the intended
migration.

## Coverage

| upstream surface | coverage |
| --- | --- |
| `TSNE` | Construction, `prepare_initial`, and the two-phase `fit` workflow |
| `TSNEEmbedding` | `numpy.ndarray` subclass and `optimize`, including in-place and copy modes |
| `tsne.kl_divergence_bh` | Sparse attraction with exact or two-dimensional Barnes-Hut repulsion |
| `tsne.kl_divergence_fft` | Two-dimensional padded FFT particle-mesh repulsion |
| `tsne.gradient_descent` | Optimization used by the tested `TSNEEmbedding` workflows |
| `affinity.PerplexityBasedNN` | Exact/automatic sklearn neighbor search, Mojo entropy search, sparse symmetrization, `set_perplexity` |
| `affinity.MultiscaleMixture` | Mixtures over a shared nearest-neighbor graph |
| `initialization` | `random`, `pca`, and `spectral` |

The following openTSNE features are not covered: approximate neighbor backends
such as Annoy and PyNNDescent, one-dimensional FFT interpolation, cached
interpolation grids for `transform`, `PartialTSNEEmbedding`, affinity
`to_new`, precomputed KNN index classes, and embedding dimensions above two
for Barnes-Hut or FFT optimization. Exact repulsion remains available for
other dimensions through `theta=0`.

There is one deliberate numerical difference. openTSNE's quadtree insertion
can retain a point in an internal node, where exact traversal cannot see it,
when that point is within `1e-6` of the node's center of mass. This port keeps
every point in a leaf. At openTSNE's `1e-4` initialization scale, short
optimization trajectories can therefore differ by a few parts in `1e-6`;
the final objective and normal-scale Barnes-Hut gradients still agree within
the tolerances asserted in the parity suite.

## Install

The repository carries its own pinned Mojo nightly and all Python test
dependencies:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-opentsne.so`. The Python bridge also
rebuilds it when the Mojo source is newer. A packaged deployment can point
`MOJO_OPENTSNE_LIB` at an already-built shared library.

## Benchmarks

Measured with the checked-in `bench/bench.py`, using one upstream worker and
the same arrays in both implementations. The Mojo FFT uses up to eight
pocketfft workers only for grids of at least 65,536 cells:

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.

| case | mojo-opentsne | openTSNE 1.0.4 | relative |
| --- | ---: | ---: | ---: |
| PerplexityBasedNN (5k x 12, p=30) | 352.4 ms | 1200.4 ms | 3.41x faster |
| Barnes-Hut objective (8k, theta=.5) | 48.3 ms | 64.3 ms | 1.33x faster |
| Exact objective (2k, theta=0) | 25.0 ms | 244.7 ms | 9.79x faster |
| FFT interpolation objective (8k) | 21.5 ms | 30.3 ms | 1.41x faster |

Run `pixi run bench` to reproduce the table. The task takes a machine-wide
file lock before measuring.

All four measured paths are faster in this run. The objective kernels skip KL
logarithms when error evaluation is disabled, as it is during normal optimizer
iterations. The FFT path uses in-place SciPy pocketfft transforms, SIMD grid
kernels, and a closed-form gridded self-potential while retaining the denser
grid needed for bilinear particle-mesh accuracy. The tests compare the
approximate gradients, not merely whether the calls complete.

No GPU path is included. The FFT support kernels are memory-bound and remain
below roughly two floating-point operations per byte moved; the higher-intensity
FFT itself is already handled by optimized pocketfft. Moving the grids to a GPU
for the support kernels would add transfers without enough arithmetic work to
amortize them.

## How it works

The Python layer owns every allocation. C-contiguous `float64` arrays and
native `int32` or `int64` CSR index arrays cross `ctypes` as integer addresses
without index conversion when the CSR storage is already contiguous.
The exported Mojo functions use `@export("name")` with `abi("C")` and rebuild
`UnsafePointer[..., AnyOrigin[mut=True]]` values internally. No Mojo allocation
or Python object crosses the ABI.

The Barnes-Hut kernel builds a flattened quadtree in NumPy-provided scratch
buffers, then performs a stack-based force traversal for each point. `theta=0`
selects the exact all-pairs kernel. Attractive forces walk the CSR affinity
matrix directly, so dense `N x N` probabilities are never materialized.

The FFT method spreads point mass to a uniform grid with bilinear
cloud-in-cell weights. NumPy-owned complex buffers are transformed in place
by SciPy pocketfft. Mojo builds the Student-kernel fields, multiplies spectra
with SIMD, interpolates the fields back to the points, and removes each
particle's gridded self-potential before normalization. Arrays are row-major
throughout.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite reports 30 passing numerical and behavioral cases against the installed
openTSNE 1.0.4 package, including generalized degrees of freedom, transformed
points, affinity matrices, initialization, optimizer trajectories, callbacks,
multiscale affinities, quantified FFT approximation accuracy, SIMD tails, KL
error-work elision, the FFT parallel threshold, FFI input validation, and
zero-copy CSR indices.

## License

MIT
