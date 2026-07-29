"""ctypes bridge to the Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_OPENTSNE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-opentsne.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mot_positive_gradient": ([I] * 9 + [F], F),
    "mot_positive_gradient_i32": ([I] * 9 + [F], F),
    "mot_negative_exact": ([I] * 6 + [F, I], F),
    "mot_negative_bh": ([I, I, I, I, I, F, F] + [I] * 11, F),
    "mot_negative_fft": ([I] * 6 + [F] * 4, F),
    "mot_deposit_grid": ([I] * 4 + [F] * 3, None),
    "mot_make_kernel": ([I, I, F, F, I], None),
    "mot_multiply_spectrum": ([I] * 3, None),
    "mot_interpolate_field": ([I] * 5 + [F] * 3 + [I, F], F),
    "mot_scale_buffer": ([I, I, F], None),
    "mot_sum_buffer": ([I, I], F),
    "mot_perplexity_probabilities": ([I, I, I, I, F], None),
    "mot_multiscale_probabilities": ([I] * 6, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "opentsne.mojo")
    if os.environ.get("MOJO_OPENTSNE_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_OPENTSNE_LIB does not exist: {LIB}")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_LIBRARY, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _LIBRARY


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays.")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous.")
    if array.size == 0:
        raise ValueError("FFI buffers must not be empty.")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffer has a null data pointer.")
    return address
