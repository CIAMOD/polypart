"""Test cases for Hyperplane initialization and validation."""

import pytest
import numpy as np
from fractions import Fraction

from polypart.geometry import Hyperplane


class TestHyperplaneInit:
    """Test the Hyperplane class constructor and __post_init__ validation."""

    def test_valid_hyperplane_creation(self):
        """Test that valid parameters create a Hyperplane successfully."""
        # Valid case 1: simple 2D hyperplane with Fraction normal and offset
        normal = np.array([Fraction(1, 2), Fraction(-1, 3)], dtype=object)
        offset = Fraction(2, 5)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_valid_hyperplane_creation_3d(self):
        """Test valid 3D hyperplane creation."""
        normal = np.array([Fraction(1), Fraction(0), Fraction(-1, 2)], dtype=object)
        offset = Fraction(3, 4)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_valid_hyperplane_creation_zero_offset(self):
        """Test valid hyperplane with zero offset."""
        normal = np.array([Fraction(1), Fraction(1)], dtype=object)
        offset = Fraction(0)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_invalid_offset_int(self):
        """Test that int offset raises TypeError."""
        normal = np.array([Fraction(1), Fraction(0)], dtype=object)
        offset = 1  # int instead of Fraction

        with pytest.raises(TypeError, match="offset must be Fraction, got int"):
            Hyperplane(normal, offset)

    def test_invalid_offset_float(self):
        """Test that float offset raises TypeError."""
        normal = np.array([Fraction(1), Fraction(0)], dtype=object)
        offset = 1.5  # float instead of Fraction

        with pytest.raises(TypeError, match="offset must be Fraction, got float"):
            Hyperplane(normal, offset)

    def test_invalid_offset_string(self):
        """Test that string offset raises TypeError."""
        normal = np.array([Fraction(1), Fraction(0)], dtype=object)
        offset = "1/2"  # string instead of Fraction

        with pytest.raises(TypeError, match="offset must be Fraction, got str"):
            Hyperplane(normal, offset)

    def test_invalid_normal_list(self):
        """Test that list normal raises TypeError."""
        normal = [Fraction(1), Fraction(0)]  # list instead of numpy array
        offset = Fraction(1)

        with pytest.raises(
            TypeError, match="normal must be a numpy.ndarray with dtype=object"
        ):
            Hyperplane(normal, offset)

    def test_invalid_normal_numpy_wrong_dtype(self):
        """Test that numpy array with wrong dtype raises TypeError."""
        normal = np.array([1.0, 0.0])  # float64 dtype instead of object
        offset = Fraction(1)

        with pytest.raises(
            TypeError, match="normal must be a numpy.ndarray with dtype=object"
        ):
            Hyperplane(normal, offset)

    def test_invalid_normal_numpy_int_dtype(self):
        """Test that numpy array with int dtype raises TypeError."""
        normal = np.array([1, 0], dtype=int)  # int dtype instead of object
        offset = Fraction(1)

        with pytest.raises(
            TypeError, match="normal must be a numpy.ndarray with dtype=object"
        ):
            Hyperplane(normal, offset)

    def test_invalid_normal_contains_int(self):
        """Test that normal containing int instead of Fraction raises TypeError."""
        normal = np.array([Fraction(1), 0], dtype=object)  # contains int 0
        offset = Fraction(1)

        with pytest.raises(TypeError, match="normal must contain only Fractions"):
            Hyperplane(normal, offset)

    def test_invalid_normal_contains_float(self):
        """Test that normal containing float instead of Fraction raises TypeError."""
        normal = np.array([Fraction(1), 0.5], dtype=object)  # contains float 0.5
        offset = Fraction(1)

        with pytest.raises(TypeError, match="normal must contain only Fractions"):
            Hyperplane(normal, offset)

    def test_invalid_normal_contains_string(self):
        """Test that normal containing string instead of Fraction raises TypeError."""
        normal = np.array([Fraction(1), "1/2"], dtype=object)  # contains string "1/2"
        offset = Fraction(1)

        with pytest.raises(TypeError, match="normal must contain only Fractions"):
            Hyperplane(normal, offset)

    def test_invalid_normal_mixed_types(self):
        """Test that normal containing mixed invalid types raises TypeError."""
        normal = np.array([Fraction(1), 2, 3.0], dtype=object)  # mixed types
        offset = Fraction(1)

        with pytest.raises(TypeError, match="normal must contain only Fractions"):
            Hyperplane(normal, offset)

    def test_empty_normal_array(self):
        """Test that empty normal array raises TypeError."""
        normal = np.array([], dtype=object)  # empty array
        offset = Fraction(1)

        # This should raise TypeError when trying to access normal[0]
        with pytest.raises(ValueError, match="non-empty"):
            Hyperplane(normal, offset)

    def test_1d_normal_vector(self):
        """Test valid 1D hyperplane (point on a line)."""
        normal = np.array([Fraction(2)], dtype=object)
        offset = Fraction(3)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_large_dimension_normal(self):
        """Test valid high-dimensional hyperplane."""
        # 5D hyperplane
        normal = np.array(
            [
                Fraction(1, 2),
                Fraction(-1),
                Fraction(0),
                Fraction(3, 4),
                Fraction(-2, 3),
            ],
            dtype=object,
        )
        offset = Fraction(7, 8)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_negative_fractions(self):
        """Test hyperplane with negative fractions."""
        normal = np.array([Fraction(-1, 2), Fraction(3, -4)], dtype=object)
        offset = Fraction(-5, 6)

        hyperplane = Hyperplane(normal, offset)
        assert np.array_equal(hyperplane.normal, normal)
        assert hyperplane.offset == offset

    def test_both_parameters_invalid(self):
        """Test that when both parameters are invalid, offset is checked first."""
        normal = [1, 2]  # invalid list
        offset = 1.5  # invalid float

        # Should raise error about offset being invalid (checked first)
        with pytest.raises(TypeError):
            Hyperplane(normal, offset)

    def test_hyperplane_immutability(self):
        """Test that Hyperplane is immutable (frozen dataclass)."""
        normal = np.array([Fraction(1), Fraction(0)], dtype=object)
        offset = Fraction(1)

        hyperplane = Hyperplane(normal, offset)

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            hyperplane.offset = Fraction(2)

        with pytest.raises(AttributeError):
            hyperplane.normal = np.array([Fraction(0), Fraction(1)], dtype=object)
