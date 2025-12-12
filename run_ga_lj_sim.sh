#!/bin/bash
#
# Run Ga-only LJ simulation with KOKKOS acceleration
# Usage: ./run_ga_lj_sim.sh [--gpu] [--cpu]
#

set -e

# Configuration
LAMMPS_INPUT="inputs/in.ga_lj_150C"
RDF_OUTPUT="rdf.dat"
OUTPUT_DIR="outputs"
USE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --cpu)
            USE_GPU=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Ga-only LJ Simulation with KOKKOS"
echo "=========================================="
echo "Input file: $LAMMPS_INPUT"
echo "GPU acceleration: $USE_GPU"
echo ""

# Check if input file exists
if [ ! -f "$LAMMPS_INPUT" ]; then
    echo "Error: Input file not found: $LAMMPS_INPUT"
    exit 1
fi

# Determine LAMMPS executable and KOKKOS options
LAMMPS_BIN="lmp"

if ! command -v $LAMMPS_BIN &> /dev/null; then
    echo "Error: LAMMPS executable '$LAMMPS_BIN' not found."
    echo "Available commands to try:"
    echo "  - which lmp"
    echo "  - which lmp_gpu"
    echo "  - which lmp_kokkos"
    echo "  - module avail lammps"
    exit 1
fi

echo "Using LAMMPS: $(which $LAMMPS_BIN)"
echo ""

# Build KOKKOS arguments
KOKKOS_ARGS=""
if [ "$USE_GPU" = true ]; then
    echo "KOKKOS mode: GPU acceleration enabled"
    KOKKOS_ARGS="-kokkos on gpu"
else
    echo "KOKKOS mode: CPU acceleration (default)"
    KOKKOS_ARGS="-kokkos on"
fi
echo ""

# Run LAMMPS simulation
echo "Starting simulation..."
echo "Command: $LAMMPS_BIN $KOKKOS_ARGS -in $LAMMPS_INPUT"
echo "=========================================="

if $LAMMPS_BIN $KOKKOS_ARGS -in "$LAMMPS_INPUT" > simulation.log 2>&1; then
    echo "✓ Simulation completed successfully"
    echo "  Log file: simulation.log"
else
    echo "✗ Simulation failed"
    echo "  Log file: simulation.log"
    tail -50 simulation.log
    exit 1
fi

echo ""
echo "=========================================="
echo "Simulation Results"
echo "=========================================="

# Check if RDF file was created
if [ -f "$RDF_OUTPUT" ]; then
    echo "✓ RDF file created: $RDF_OUTPUT"
    num_lines=$(wc -l < "$RDF_OUTPUT")
    echo "  Lines: $num_lines"
else
    echo "✗ RDF file not found: $RDF_OUTPUT"
    exit 1
fi

# Check if trajectory file was created
if [ -f "dump.ga.lj.lammpstrj" ]; then
    echo "✓ Trajectory file created: dump.ga.lj.lammpstrj"
    num_frames=$(grep "ITEM: TIMESTEP" dump.ga.lj.lammpstrj | wc -l)
    echo "  Frames: $num_frames"
fi

# Check if final structure was written
if [ -f "outputs/final_structure_ga_lj.data" ]; then
    echo "✓ Final structure saved: outputs/final_structure_ga_lj.data"
fi

echo ""
echo "=========================================="
echo "Next step: Analyze S(Q)"
echo "=========================================="
echo ""
echo "Run the analysis script:"
echo "  python scripts/run_ga_lj_analyze_sq.py"
echo ""
echo "This will generate:"
echo "  - outputs/ga_lj_sq.png (g(r) and S(Q) plots)"
echo "  - outputs/ga_lj_sq.txt (S(Q) data)"
echo ""
