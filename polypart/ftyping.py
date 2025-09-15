from __future__ import annotations
from fractions import Fraction
from typing import TypeAlias, Iterable, Union
import numpy as np


# Include numpy scalar numeric types so they are accepted as "number-like"
NumberLike = Union[int, float, Fraction, np.integer, np.floating]

FractionVector: TypeAlias = np.ndarray
"""A 1D numpy array of dtype=object, containing Fraction objects. Shape: (d,)"""

FractionMatrix: TypeAlias = np.ndarray
"""A 2D numpy array of dtype=object, containing Fraction objects. Shape: (n, d)"""


def to_fraction(x: NumberLike, *, max_denominator: int = 10**8) -> Fraction:
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
    # numpy integer scalar (e.g. np.int64) as well as Python int
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x), 1)
    # numpy float scalar (e.g. np.float64) as well as Python float
    if isinstance(x, (float, np.floating)):
        return Fraction(float(x)).limit_denominator(max_denominator)
    raise TypeError(f"Cannot convert type {type(x)!r} to Fraction")


def as_fraction_matrix(rows: Iterable[Iterable[NumberLike]]) -> FractionMatrix:
    """Create a 2-D object-dtype numpy array of Fractions.

    Args:
        rows: iterable of rows, each an iterable of number-like values.

    Returns:
        2-D numpy array (dtype=object) of Fraction objects.
    """
    data = [[to_fraction(v) for v in row] for row in rows]
    return np.array(data, dtype=object)


def as_fraction_vector(vals: Iterable[NumberLike]) -> FractionVector:
    """Create a 1-D object-dtype numpy array of Fractions.

    Args:
        vals: iterable of number-like values.

    Returns:
        1-D numpy array (dtype=object) of Fraction objects.
    """
    return np.array([to_fraction(v) for v in vals], dtype=object)
