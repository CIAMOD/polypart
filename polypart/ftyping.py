"""Type aliases and utility functions for handling number-like values and
fractions.
"""

from __future__ import annotations

from fractions import Fraction as SlowFraction
from typing import Iterable, Literal, TypeAlias, Union

import numpy as np
from gmpy2 import mpq

Fraction: TypeAlias = mpq
"""Rational number type using gmpy2.mpq for better performance."""

# Include numpy scalar numeric types so they are accepted as "number-like"
NumberLike = Union[int, float, SlowFraction, Fraction, np.integer, np.floating]

# Strategy type for hyperplane selection
SplitStrategy: TypeAlias = Literal["random", "v-entropy"]

FractionVector: TypeAlias = np.ndarray
"""A 1D numpy array of dtype=object, containing Fraction objects. Shape: (d,)"""

FractionMatrix: TypeAlias = np.ndarray
"""A 2D numpy array of dtype=object, containing Fraction objects. Shape: (n, d)"""


def to_fraction(x: NumberLike) -> Fraction:
    """Convert a number-like value to a Fraction.

    Args:
        x: int/float/Fraction or numpy numeric scalar.
        max_denominator: maximum denominator for float conversion.

    Returns:
        Fraction representation of ``x``.

    Notes:
        Floats are converted via ``Fraction(float(x))`` and may lose
        precision. Integers and Fractions are returned exactly.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, SlowFraction):
        return Fraction(int(x.numerator), int(x.denominator))
    # numpy integer scalar (e.g. np.int64) as well as Python int
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    # numpy float scalar (e.g. np.float64) as well as Python float
    if isinstance(x, (float, np.floating)):
        return Fraction(float(x))
    raise TypeError(f"Cannot convert type {type(x)!r} to Fraction")


def as_fraction_matrix(rows: Iterable[Iterable[NumberLike]]) -> FractionMatrix:
    """Create a 2-D object-dtype numpy array of Fractions.

    Args:
        rows: iterable of rows, each an iterable of number-like values.

    Returns:
        2-D numpy array (dtype=object) of Fraction objects.
    """
    # Let numpy materialise the 2-D object array
    arr = np.array(list(rows), dtype=object)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D input, got shape {arr.shape}")

    # Elementwise conversion to Fraction using a ufunc
    to_frac_ufunc = np.frompyfunc(to_fraction, 1, 1)
    frac_arr = to_frac_ufunc(arr)
    # frompyfunc already yields dtype=object
    return frac_arr


def as_fraction_vector(
    vals: Iterable[NumberLike], decimals: int = None
) -> FractionVector:
    """Create a 1-D object-dtype numpy array of Fractions.

    Args:
        vals: iterable of number-like values.
        decimals: if set, limit denominators to 10**decimals.

    Returns:
        1-D numpy array (dtype=object) of Fraction objects.
    """
    # Let numpy materialise the 1-D object array
    arr = np.array(list(vals), dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D input, got shape {arr.shape}")

    # Elementwise conversion to Fraction using a ufunc
    to_frac_ufunc = np.frompyfunc(to_fraction, 1, 1)
    frac_arr = to_frac_ufunc(arr)
    # frompyfunc already yields dtype=object
    if decimals is not None:
        limit = 10**decimals
        for i in range(frac_arr.shape[0]):
            frac_arr[i] = frac_arr[i].limit_denominator(limit)
    return frac_arr
