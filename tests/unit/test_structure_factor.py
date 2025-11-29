"""
Unit tests for structure factor calculations.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from epsr.core.structure_factor import (
    StructureFactor,
    calculate_structure_factor,
    calculate_F_Q,
    calculate_T_r
)


class TestStructureFactor:
    """Test StructureFactor class."""

    def test_initialization(self):
        """Test StructureFactor initialization."""
        sf = StructureFactor(rho=0.042)
        assert sf.rho == 0.042
        assert sf.r_max == 30.0
        assert sf.n_points == 1000

    def test_g_to_S_simple(self):
        """Test g(r) to S(Q) transformation with simple case."""
        sf = StructureFactor(rho=0.042)

        # Simple test case: uniform g(r) = 1 should give S(Q) = 1
        r = np.linspace(0.1, 10.0, 100)
        g = np.ones_like(r)
        Q = np.array([1.0, 2.0, 3.0])

        S = sf.g_to_S(r, g, Q)

        # For g(r) = 1, S(Q) should be close to 1
        assert np.allclose(S, 1.0, atol=0.1)

    def test_calculate_F_Q(self):
        """Test F(Q) = Q[S(Q) - 1] calculation."""
        Q = np.array([1.0, 2.0, 3.0])
        S = np.array([1.1, 1.2, 1.3])

        F = calculate_F_Q(Q, S)

        expected = Q * (S - 1.0)
        assert np.allclose(F, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
