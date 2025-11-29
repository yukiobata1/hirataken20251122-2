"""
Unit tests for metrics calculations.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from epsr.analysis.metrics import (
    calculate_chi_squared,
    calculate_r_factor,
    calculate_residuals,
    calculate_rmsd
)


class TestMetrics:
    """Test goodness-of-fit metrics."""

    def test_chi_squared_perfect_fit(self):
        """Test chi-squared for perfect fit."""
        y_sim = np.array([1.0, 2.0, 3.0, 4.0])
        y_exp = np.array([1.0, 2.0, 3.0, 4.0])

        chi2 = calculate_chi_squared(y_sim, y_exp, sigma=0.1)

        assert chi2 == 0.0

    def test_chi_squared_with_error(self):
        """Test chi-squared with known error."""
        y_sim = np.array([1.1, 2.1, 3.1, 4.1])
        y_exp = np.array([1.0, 2.0, 3.0, 4.0])
        sigma = 0.1

        chi2 = calculate_chi_squared(y_sim, y_exp, sigma)

        # χ² = Σ(0.1²/0.1²) = 4
        assert np.isclose(chi2, 4.0)

    def test_r_factor_perfect_fit(self):
        """Test R-factor for perfect fit."""
        y_sim = np.array([1.0, 2.0, 3.0])
        y_exp = np.array([1.0, 2.0, 3.0])

        r = calculate_r_factor(y_sim, y_exp)

        assert r == 0.0

    def test_r_factor_with_error(self):
        """Test R-factor calculation."""
        y_sim = np.array([1.1, 2.0, 2.9])
        y_exp = np.array([1.0, 2.0, 3.0])

        r = calculate_r_factor(y_sim, y_exp)

        # R = (0.1 + 0.0 + 0.1) / (1.0 + 2.0 + 3.0) = 0.2/6.0
        expected = 0.2 / 6.0
        assert np.isclose(r, expected)

    def test_residuals(self):
        """Test residuals calculation."""
        y_sim = np.array([1.1, 2.2, 3.3])
        y_exp = np.array([1.0, 2.0, 3.0])

        residuals = calculate_residuals(y_sim, y_exp)

        expected = np.array([0.1, 0.2, 0.3])
        assert np.allclose(residuals, expected)

    def test_rmsd(self):
        """Test RMSD calculation."""
        y_sim = np.array([1.1, 2.0, 2.9])
        y_exp = np.array([1.0, 2.0, 3.0])

        rmsd = calculate_rmsd(y_sim, y_exp)

        # RMSD = sqrt((0.1² + 0² + 0.1²)/3)
        expected = np.sqrt((0.01 + 0.0 + 0.01) / 3)
        assert np.isclose(rmsd, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
