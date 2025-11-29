#!/usr/bin/env python3
"""
Basic tests for EPSR package (no pytest required).
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from epsr import EPSREngine
        print("  ✓ EPSREngine")
    except Exception as e:
        print(f"  ✗ EPSREngine: {e}")
        return False

    try:
        from epsr.core.structure_factor import StructureFactor
        print("  ✓ StructureFactor")
    except Exception as e:
        print(f"  ✗ StructureFactor: {e}")
        return False

    try:
        from epsr.core.potential import EmpiricalPotential
        print("  ✓ EmpiricalPotential")
    except Exception as e:
        print(f"  ✗ EmpiricalPotential: {e}")
        return False

    try:
        from epsr.analysis.metrics import calculate_chi_squared
        print("  ✓ Metrics")
    except Exception as e:
        print(f"  ✗ Metrics: {e}")
        return False

    try:
        from epsr.io.experimental import load_experimental_data
        print("  ✓ IO modules")
    except Exception as e:
        print(f"  ✗ IO modules: {e}")
        return False

    return True


def test_structure_factor():
    """Test structure factor calculations."""
    print("\nTesting StructureFactor...")

    from epsr.core.structure_factor import StructureFactor, calculate_F_Q, calculate_T_r

    # Test initialization
    sf = StructureFactor(rho=0.042)
    assert sf.rho == 0.042, "Density not set correctly"
    print("  ✓ Initialization")

    # Test F(Q) calculation
    Q = np.array([1.0, 2.0, 3.0])
    S = np.array([1.1, 1.2, 1.3])
    F = calculate_F_Q(Q, S)
    expected = Q * (S - 1.0)
    assert np.allclose(F, expected), "F(Q) calculation failed"
    print("  ✓ F(Q) calculation")

    # Test T(r) calculation
    r = np.array([1.0, 2.0, 3.0])
    g = np.array([0.5, 1.5, 2.0])
    rho = 0.042
    T = calculate_T_r(r, g, rho)
    expected = 4.0 * np.pi * rho * r * (g - 1.0)
    assert np.allclose(T, expected), "T(r) calculation failed"
    print("  ✓ T(r) calculation")

    # Test g to S transform (simple case)
    r = np.linspace(0.1, 10.0, 100)
    g = np.ones_like(r)  # Uniform g(r) = 1
    Q = np.array([1.0, 2.0, 3.0])
    S = sf.g_to_S(r, g, Q)
    # For g(r) = 1, S(Q) should be close to 1
    assert np.allclose(S, 1.0, atol=0.1), "g to S transform failed"
    print("  ✓ g(r) → S(Q) transform")

    return True


def test_metrics():
    """Test goodness-of-fit metrics."""
    print("\nTesting metrics...")

    from epsr.analysis.metrics import (
        calculate_chi_squared,
        calculate_r_factor,
        calculate_residuals,
        calculate_rmsd
    )

    # Test chi-squared (perfect fit)
    y_sim = np.array([1.0, 2.0, 3.0, 4.0])
    y_exp = np.array([1.0, 2.0, 3.0, 4.0])
    chi2 = calculate_chi_squared(y_sim, y_exp, sigma=0.1)
    assert chi2 == 0.0, "Chi-squared for perfect fit should be 0"
    print("  ✓ Chi-squared (perfect fit)")

    # Test chi-squared (with error)
    y_sim = np.array([1.1, 2.1, 3.1, 4.1])
    y_exp = np.array([1.0, 2.0, 3.0, 4.0])
    chi2 = calculate_chi_squared(y_sim, y_exp, sigma=0.1)
    assert np.isclose(chi2, 4.0), "Chi-squared calculation failed"
    print("  ✓ Chi-squared (with error)")

    # Test R-factor (perfect fit)
    y_sim = np.array([1.0, 2.0, 3.0])
    y_exp = np.array([1.0, 2.0, 3.0])
    r = calculate_r_factor(y_sim, y_exp)
    assert r == 0.0, "R-factor for perfect fit should be 0"
    print("  ✓ R-factor (perfect fit)")

    # Test residuals
    y_sim = np.array([1.1, 2.2, 3.3])
    y_exp = np.array([1.0, 2.0, 3.0])
    residuals = calculate_residuals(y_sim, y_exp)
    expected = np.array([0.1, 0.2, 0.3])
    assert np.allclose(residuals, expected), "Residuals calculation failed"
    print("  ✓ Residuals")

    # Test RMSD
    y_sim = np.array([1.1, 2.0, 2.9])
    y_exp = np.array([1.0, 2.0, 3.0])
    rmsd = calculate_rmsd(y_sim, y_exp)
    expected = np.sqrt((0.01 + 0.0 + 0.01) / 3)
    assert np.isclose(rmsd, expected), "RMSD calculation failed"
    print("  ✓ RMSD")

    return True


def test_potential():
    """Test empirical potential calculations."""
    print("\nTesting EmpiricalPotential...")

    from epsr.core.potential import EmpiricalPotential, calculate_scattering_weights

    # Test initialization
    r_grid = np.linspace(2.0, 12.0, 200)
    ep = EmpiricalPotential(
        r_grid=r_grid,
        temperature=423.15,
        rho=0.042
    )
    assert len(ep.U_ep) == len(r_grid), "Potential grid size mismatch"
    print("  ✓ Initialization")

    # Test scattering weights calculation
    composition = {'Ga': 0.858, 'In': 0.142}
    scattering_lengths = {'Ga': 7.288, 'In': 4.061}
    weights = calculate_scattering_weights(composition, scattering_lengths)

    assert 'GaGa' in weights, "GaGa weight missing"
    assert 'InIn' in weights, "InIn weight missing"
    assert 'GaIn' in weights, "GaIn weight missing"

    # Weights should sum to 1
    total_weight = sum(weights.values())
    assert np.isclose(total_weight, 1.0, atol=0.01), f"Weights should sum to 1, got {total_weight}"
    print("  ✓ Scattering weights")

    # Test potential update
    g_sim = np.ones_like(r_grid)
    g_exp = np.ones_like(r_grid) * 1.1

    U_new = ep.update_from_rdf(g_sim, g_exp, alpha=0.1)

    # Potential should change
    assert not np.allclose(U_new, 0.0), "Potential should have been updated"
    print("  ✓ Potential update")

    return True


def test_rdf():
    """Test RDF analysis."""
    print("\nTesting RDF analysis...")

    from epsr.analysis.rdf import RadialDistributionFunction, calculate_coordination_number

    # Create test RDF
    r = np.linspace(0.5, 10.0, 100)
    g = np.ones_like(r)
    g += 2.0 * np.exp(-((r - 2.7)**2) / (2 * 0.3**2))  # Peak at 2.7 Å

    rdf = RadialDistributionFunction(r, g, rho=0.042)

    # Test peak finding
    r_peak = rdf.first_peak_position(r_min=2.0, r_max=4.0)
    assert 2.5 < r_peak < 3.0, f"Peak should be around 2.7, got {r_peak}"
    print("  ✓ Peak finding")

    # Test coordination number
    coord_num = calculate_coordination_number(r, g, rho=0.042, r_max=3.5)
    assert coord_num > 0, "Coordination number should be positive"
    print("  ✓ Coordination number")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("EPSR Package Tests")
    print("=" * 60)

    all_passed = True

    tests = [
        ("Imports", test_imports),
        ("StructureFactor", test_structure_factor),
        ("Metrics", test_metrics),
        ("EmpiricalPotential", test_potential),
        ("RDF Analysis", test_rdf),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
            all_passed = False

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:25s} {status}")

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
