#!/bin/bash
# Quick start script for EPSR with H100 GPU acceleration
# This script runs the full EPSR workflow optimized for NVIDIA H100

set -e  # Exit on error

echo "================================================"
echo "  EPSR for EGaIn with H100 GPU Acceleration"
echo "================================================"
echo ""

# Check if LAMMPS with Kokkos is available
if ! command -v lmp &> /dev/null; then
    echo "❌ Error: lmp command not found"
    echo ""
    echo "Please install LAMMPS with Kokkos support first:"
    echo "  bash scripts/build_lammps_kokkos.sh"
    echo ""
    exit 1
fi

# Check if LAMMPS has Kokkos support
if ! lmp -help | grep -q "KOKKOS"; then
    echo "⚠️  Warning: LAMMPS may not have KOKKOS support"
    echo "   Consider rebuilding with: bash scripts/build_lammps_kokkos.sh"
    echo ""
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU Status:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found. Cannot verify GPU availability"
    echo ""
fi

# Check if initial structure exists
if [ ! -f "inputs/initial_structure.data" ]; then
    echo "📦 Creating initial structure..."
    python3 scripts/create_initial_structure.py
    echo ""
fi

# Check if experimental data exists
if [ ! -f "data/g_exp.dat" ]; then
    echo "❌ Error: Experimental data not found at data/g_exp.dat"
    echo ""
    echo "Please provide experimental g(r) data:"
    echo "  1. Download from Supporting Information of Amon et al. (2023)"
    echo "  2. Extract and save to data/g_exp.dat"
    echo "  3. Format: 2 columns (r in Å, g(r))"
    echo ""
    exit 1
fi

# Check if empirical potential tables exist
if [ ! -f "data/ep_GaGa.table" ] || [ ! -f "data/ep_InIn.table" ] || [ ! -f "data/ep_GaIn.table" ]; then
    echo "📦 Initializing empirical potential tables..."
    python3 scripts/update_ep.py
    echo ""
fi

# Create outputs directory
mkdir -p outputs

echo "================================================"
echo "Starting EPSR simulation..."
echo "================================================"
echo ""

# Option to test with LJ-only first
if [ "$1" == "--test-lj" ]; then
    echo "🧪 Test mode: Running LJ-only simulation first"
    echo ""
    lmp -k on g 1 -sf kk -in inputs/in.egain_lj_H100 -log outputs/test_lj_H100.log
    echo ""
    echo "✅ Test simulation completed. Check outputs/test_lj_H100.log"
    echo "   RDF saved to rdf.dat - compare with data/g_exp.dat"
    echo ""
    echo "To run full EPSR, execute:"
    echo "  bash scripts/run_epsr_H100.sh"
    exit 0
fi

# Run main EPSR loop
echo "🚀 Starting EPSR iterations (this may take a while)..."
echo ""

python3 scripts/main_epsr.py

echo ""
echo "================================================"
echo "  ✅ EPSR Complete!"
echo "================================================"
echo ""
echo "Results:"
echo "  📊 Plots: outputs/epsr_iter###.png"
echo "  📊 Final: outputs/epsr_final.png"
echo "  💾 Data:  outputs/final_ep.npz"
echo "  📝 Logs:  outputs/lammps_iter###.log"
echo ""
echo "To analyze results:"
echo "  python3 -c 'import numpy as np; d=np.load(\"outputs/final_ep.npz\"); print(f\"Final χ²: {d[\"chi2_history\"][-1]:.6f}\")'"
echo ""
