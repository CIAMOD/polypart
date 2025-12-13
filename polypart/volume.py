import subprocess
import tempfile
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np

from .ftyping import Fraction, FractionMatrix, FractionVector


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return abs(a) or abs(b)
    return abs(a * b) // gcd(a, b)


def _row_to_integer_inequality(a_row: FractionVector, b_i: Fraction) -> list[int]:
    """
    Given a single inequality a_row · x <= b_i with rational entries,
    return integer coefficients [c1,...,cd,c0] representing

        c1 x1 + ... + cd xd + c0 >= 0

    in Normaliz's inhom_inequalities format.

    For A x <= b we use:
        A' x <= b'  with integer A', b'
        -A' x >= -b'
    and Normaliz row (-A', b') encodes ξ·x >= η with ξ = -A', η = -b'.
    """
    a_row = list(a_row)
    coeffs = a_row + [b_i]

    # common denominator over all coefficients and RHS
    den_lcm = reduce(_lcm, (c.denominator for c in coeffs), 1)

    ints = []
    for c in coeffs:
        num, den = c.numerator, c.denominator
        ints.append(num * (den_lcm // den))

    a_int = ints[:-1]
    b_int = ints[-1]

    # -A' x >= -b'   ->  row (-A', b')
    return [-c for c in a_int] + [b_int]


def write_normaliz_input(A: FractionMatrix, b: FractionVector, path: Path):
    """
    Write a Normaliz input file describing the polyhedron P = { x in R^d : A x <= b }
    with A and b possibly rational (mpq/Fraction/int)
    so that Normaliz computes the volume.

    The file uses:
        amb_space d
        inhom_inequalities m
        <rows of (-A', b')>
        Volume
    """
    m, d = A.shape
    assert b.shape[0] == m

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"amb_space {d}\n")
        f.write(f"inhom_inequalities {m}\n")
        for i in range(m):
            row = _row_to_integer_inequality(A[i, :], b[i])
            if len(row) != d + 1:  # safety: must be d+1 entries
                raise ValueError(
                    f"Row {i} has wrong length {len(row)} (expected {d + 1})"
                )
            f.write(" ".join(str(c) for c in row) + "\n")
        f.write("Volume\n")


def run_normaliz_input(in_file: Path, normaliz_exe: str = "normaliz") -> Path:
    """
    Run Normaliz on the given .in file.

    Returns the path to the corresponding .out file.
    """
    in_file = Path(in_file)
    # Run Normaliz in the same directory as the input file so output files
    # (.out, etc.) are created next to the input and can be cleaned up.
    try:
        subprocess.run([normaliz_exe, str(in_file)], check=True, cwd=in_file.parent)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Normaliz executable '{normaliz_exe}' not found. "
            "Please ensure Normaliz is installed and in your system PATH."
        )
    out_file = in_file.with_suffix(".out")
    if not out_file.exists():
        raise FileNotFoundError(f"Normaliz did not create {out_file}")
    return out_file


def _extract_volume_from_out(out_file: Path) -> Fraction:
    """
    volume (Euclidean) = Vol

    This is the Z^d-normalised volume of the Ehrhart cone.
    """
    out_file = Path(out_file)
    text = out_file.read_text(encoding="utf-8")

    for line in text.splitlines():
        if line.startswith("volume (Euclidean) ="):
            parts = line.split("=")
            vol_str = parts[1].strip()
            return Fraction(vol_str)

    raise ValueError(f"No 'volume (Euclidean) =' line found in {out_file}. ")


def volume_nmz(
    A: FractionMatrix, b: FractionVector, normaliz_exe: str = "normaliz"
) -> Fraction:
    """
    Compute the Euclidean volume of polytope P = { x in R^d : A x <= b }
    using Normaliz via Ehrhart multiplicity.

    Returns:
        Volume as a Fraction (exact rational).
    """
    # Use a temporary directory for Normaliz input/output files
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = Path(tmpdir) / "polytope.in"
        write_normaliz_input(A, b, in_file)
        out_file = run_normaliz_input(in_file, normaliz_exe=normaliz_exe)
        vol = _extract_volume_from_out(out_file)
        return vol
