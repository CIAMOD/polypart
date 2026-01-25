"""Volume computation using Normaliz."""

import shutil
import subprocess
import tempfile
from functools import reduce
from math import gcd
from pathlib import Path

from polypart.core.typing import Fraction, FractionMatrix, FractionVector


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return abs(a) or abs(b)
    return abs(a * b) // gcd(a, b)


def _row_to_integer_inequality(a_row: FractionVector, b_i: Fraction) -> list[int]:
    """Convert rational inequality to integer form for Normaliz."""
    a_row_list = list(a_row)
    coeffs = a_row_list + [b_i]
    den_lcm = reduce(_lcm, (c.denominator for c in coeffs), 1)

    ints = []
    for c in coeffs:
        num, den = c.numerator, c.denominator
        ints.append(num * (den_lcm // den))

    a_int = ints[:-1]
    b_int = ints[-1]
    return [-c for c in a_int] + [b_int]


def write_normaliz_input(A: FractionMatrix, b: FractionVector, path: Path) -> None:
    """Write Normaliz input file for polytope P = { x : A x <= b }."""
    m, d = A.shape
    assert b.shape[0] == m

    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"amb_space {d}\n")
        f.write(f"inhom_inequalities {m}\n")
        for i in range(m):
            row = _row_to_integer_inequality(A[i, :], b[i])
            if len(row) != d + 1:
                raise ValueError(
                    f"Row {i} has wrong length {len(row)} (expected {d + 1})"
                )
            f.write(" ".join(str(c) for c in row) + "\n")
        f.write("Volume\n")


def run_normaliz_input(in_file: Path, normaliz_exe: str = "normaliz") -> Path:
    """Run Normaliz on input file, return output file path."""
    if shutil.which(normaliz_exe) is None:
        raise FileNotFoundError(
            f"Normaliz executable '{normaliz_exe}' not found. "
            "Ensure Normaliz is installed and in PATH."
        )

    in_file = Path(in_file)
    try:
        subprocess.run([normaliz_exe, str(in_file)], check=True, cwd=in_file.parent)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Normaliz failed with exit code {e.returncode}")

    out_file = in_file.with_suffix(".out")
    if not out_file.exists():
        raise FileNotFoundError(f"Normaliz did not create {out_file}")
    return out_file


def _extract_volume_from_out(out_file: Path) -> Fraction:
    """Extract Euclidean volume from Normaliz output."""
    out_file = Path(out_file)
    text = out_file.read_text(encoding="utf-8")

    for line in text.splitlines():
        if line.startswith("volume (Euclidean) ="):
            parts = line.split("=")
            vol_str = parts[1].strip()
            return Fraction(vol_str)

    raise ValueError(f"No 'volume (Euclidean) =' line found in {out_file}")


def volume_nmz(
    A: FractionMatrix, b: FractionVector, normaliz_exe: str = "normaliz"
) -> Fraction:
    """Compute Euclidean volume of polytope P = { x : A x <= b } using Normaliz."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = Path(tmpdir) / "polytope.in"
        write_normaliz_input(A, b, in_file)
        out_file = run_normaliz_input(in_file, normaliz_exe=normaliz_exe)
        return _extract_volume_from_out(out_file)
