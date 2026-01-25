"""Type aliases and utilities for handling rational arithmetic."""

from __future__ import annotations

from typing import Iterable, Literal, TypeAlias, Union

import numpy as np
from gmpy2 import mpq

Fraction: TypeAlias = mpq
NumberLike: TypeAlias = Union[int, float, Fraction, np.integer, np.floating]
SplitStrategy: TypeAlias = Literal["random", "v-entropy"]
FractionVector: TypeAlias = np.ndarray  # Shape [d], dtype=object
FractionMatrix: TypeAlias = np.ndarray  # Shape [n, d], dtype=object


def to_fraction(x: NumberLike) -> Fraction:
    """Convert a number-like value to a Fraction."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    if isinstance(x, (float, np.floating)):
        return Fraction(float(x))
    raise TypeError(f"Cannot convert type {type(x)!r} to Fraction")


def as_fraction_matrix(rows: Iterable[Iterable[NumberLike]]) -> FractionMatrix:
    """Create a 2-D object-dtype numpy array of Fractions."""
    arr = np.array(list(rows), dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D input, got shape {arr.shape}")
    to_frac_ufunc = np.frompyfunc(to_fraction, 1, 1)
    return to_frac_ufunc(arr)


def as_fraction_vector(
    vals: Iterable[NumberLike], decimals: int | None = None
) -> FractionVector:
    """Create a 1-D object-dtype numpy array of Fractions."""
    arr = np.array(list(vals), dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input, got shape {arr.shape}")
    to_frac_ufunc = np.frompyfunc(to_fraction, 1, 1)
    frac_arr = to_frac_ufunc(arr)
    if decimals is not None:
        limit = 10**decimals
        for i in range(frac_arr.shape[0]):
            frac_arr[i] = frac_arr[i].limit_denominator(limit)
    return frac_arr


def is_fraction_vector(
    arr: np.ndarray,
) -> bool:
    """Check if a numpy array is a 1-D array of Fractions."""
    return type(arr) is np.ndarray and type(arr[0]) is Fraction


def is_fraction_matrix(
    arr: np.ndarray,
) -> bool:
    """Check if a numpy array is a 2-D array of Fractions."""
    return type(arr) is np.ndarray and arr.ndim == 2 and type(arr[0, 0]) is Fraction
