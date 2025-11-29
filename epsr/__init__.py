"""
EPSR - Empirical Potential Structure Refinement

A modern implementation of the EPSR algorithm for refining
atomic structures against experimental scattering data.

Based on the work of A.K. Soper and inspired by the Dissolve software.

References
----------
- Soper, A. K. (1996). Chem. Phys., 202, 295-306.
- Soper, A. K. (2005). Phys. Rev. B, 72, 104204.
- Youngs, T.G.A. et al. (2019). Mol. Phys., 117:22, 3464-3477. (Dissolve)
"""

__version__ = "1.0.0"
__author__ = "EPSR Implementation Team"

from epsr.core.epsr_engine import EPSREngine
from epsr.core.potential import EmpiricalPotential
from epsr.core.structure_factor import StructureFactor
from epsr.analysis.rdf import RadialDistributionFunction
from epsr.analysis.metrics import calculate_chi_squared, calculate_r_factor

__all__ = [
    "EPSREngine",
    "EmpiricalPotential",
    "StructureFactor",
    "RadialDistributionFunction",
    "calculate_chi_squared",
    "calculate_r_factor",
]
