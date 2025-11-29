#!/usr/bin/env python3
"""
Run EPSR refinement for EGaIn system.

This is the main entry point for running EPSR simulations using the
new modular implementation.

Example usage:
    python scripts/run_epsr.py
    python scripts/run_epsr.py --gpu --max-iter 30
"""

import sys
import os
import argparse

# Add parent directory to path to import epsr package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from epsr import EPSREngine
from epsr.core.epsr_engine import EPSRConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run EPSR refinement for EGaIn system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # File paths
    parser.add_argument(
        '--exp-data',
        default='data/g_exp_cleaned.dat',
        help='Experimental g(r) data file'
    )
    parser.add_argument(
        '--lammps-input',
        default='inputs/in.egain_epsr_H100',
        help='LAMMPS input script'
    )
    parser.add_argument(
        '--rdf-output',
        default='rdf.dat',
        help='RDF output file from LAMMPS'
    )

    # System parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=423.15,
        help='Temperature (K)'
    )
    parser.add_argument(
        '--density',
        type=float,
        default=0.042,
        help='Number density (atoms/Å³)'
    )

    # EPSR parameters
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Maximum iterations'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.3,
        help='Learning rate'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=250.0,
        help='Convergence tolerance (χ²)'
    )
    parser.add_argument(
        '--method',
        choices=['simple', 'momentum', 'nesterov'],
        default='momentum',
        help='Update method'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.9,
        help='Momentum coefficient'
    )
    parser.add_argument(
        '--max-amplitude',
        type=float,
        default=3.0,
        help='Maximum potential amplitude (kcal/mol)'
    )

    # GPU options
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU acceleration (Kokkos)'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID'
    )

    # Output options
    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory for potential tables'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )

    args = parser.parse_args()

    # EGaIn composition (eutectic)
    composition = {'Ga': 0.858, 'In': 0.142}

    # Neutron scattering lengths (fm)
    scattering_lengths = {'Ga': 7.288, 'In': 4.061}

    # Create configuration
    config = EPSRConfig(
        temperature=args.temperature,
        composition=composition,
        scattering_lengths=scattering_lengths,
        density=args.density,
        max_iterations=args.max_iter,
        learning_rate=args.alpha,
        convergence_tol=args.tolerance,
        method=args.method,
        momentum_beta=args.beta,
        max_amplitude=args.max_amplitude,
        use_gpu=args.gpu,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        verbose=not args.quiet
    )

    # Check if input files exist
    if not os.path.exists(args.exp_data):
        print(f"Error: Experimental data file not found: {args.exp_data}")
        sys.exit(1)

    if not os.path.exists(args.lammps_input):
        print(f"Error: LAMMPS input file not found: {args.lammps_input}")
        sys.exit(1)

    # If using CPU-only LAMMPS, check for alternative input file
    if not args.gpu:
        cpu_input = args.lammps_input.replace('_H100', '')
        if os.path.exists(cpu_input):
            args.lammps_input = cpu_input
            print(f"Using CPU input file: {cpu_input}")

    # Create EPSR engine
    engine = EPSREngine(config, args.exp_data)

    # Run refinement
    try:
        results = engine.run(
            lammps_input=args.lammps_input,
            rdf_output=args.rdf_output,
            save_interval=1
        )

        # Print summary
        if results['converged']:
            print("\n✓ EPSR refinement converged successfully!")
        else:
            print("\n✗ EPSR refinement did not converge.")
            print("  Consider adjusting parameters (learning rate, max iterations, etc.)")

        sys.exit(0 if results['converged'] else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
