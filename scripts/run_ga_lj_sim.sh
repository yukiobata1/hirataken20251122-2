#!/bin/bash
#
# Run Ga-only LJ simulation with KOKKOS acceleration
# Usage: ./run_ga_lj_sim.sh [--gpu] [--cpu]
#

set -e

echo "================================================"
echo "  Gallium LJ Simulation with KOKKOS"
echo "================================================"
echo ""

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
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$LAMMPS_INPUT" ]; then
    echo "❌ Error: Input file '$LAMMPS_INPUT' not found"
    exit 1
fi

echo "📄 Input file: $LAMMPS_INPUT"
echo ""

# Check for LAMMPS with KOKKOS
LAMMPS_BIN="lmp"

if ! command -v $LAMMPS_BIN &> /dev/null; then
    echo "❌ Error: LAMMPS executable '$LAMMPS_BIN' not found"
    exit 1
fi

if ! lmp -help 2>/dev/null | grep -q "KOKKOS"; then
    echo "❌ Error: LAMMPS not compiled with KOKKOS support"
    exit 1
fi

echo "✅ LAMMPS with KOKKOS found: $(which $LAMMPS_BIN)"
echo ""

# Handle GPU mode
if [ "$USE_GPU" = true ]; then
    echo "🎮 GPU Mode: Checking GPU availability..."

    if ! command -v nvidia-smi &> /dev/null; then
        echo "❌ Error: nvidia-smi not found. Is CUDA installed?"
        exit 1
    fi

    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/   /'
    echo ""

    export CUDA_VISIBLE_DEVICES=0
    KOKKOS_ARGS="-k on g 1 -sf kk"
    echo "🚀 Using GPU with 1 device"
else
    echo "💻 CPU Mode: Using CPU threads"
    KOKKOS_ARGS="-k on"
fi
echo ""

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="$OUTPUT_DIR/output_ga_lj_${TIMESTAMP}.log"

echo "📊 Log file: $OUTPUT_LOG"
echo ""

# Optional: Start GPU monitoring if in GPU mode
MONITOR_PID=""
if [ "$USE_GPU" = true ]; then
    echo "📈 Starting GPU monitor in background..."
    watch -n 2 nvidia-smi > "$OUTPUT_DIR/gpu_monitor_${TIMESTAMP}.log" 2>&1 &
    MONITOR_PID=$!
    echo ""
fi

# Run LAMMPS simulation
echo "🚀 Starting simulation..."
echo "Command: $LAMMPS_BIN $KOKKOS_ARGS -in $LAMMPS_INPUT"
echo "================================================"

if time $LAMMPS_BIN $KOKKOS_ARGS -in "$LAMMPS_INPUT" > "$OUTPUT_LOG" 2>&1; then
    # Stop GPU monitor if running
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi

    echo ""
    echo "✅ Simulation completed successfully"
    echo "   Log file: $OUTPUT_LOG"
    echo ""

    # Extract performance info
    echo "📈 Performance Summary:"
    grep "Loop time of " "$OUTPUT_LOG" | tail -1 || echo "   (No performance summary found)"
    echo ""
else
    # Stop GPU monitor if running
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi

    echo ""
    echo "❌ Simulation failed"
    echo "   Log file: $OUTPUT_LOG"
    echo ""
    echo "Last 50 lines of log:"
    tail -50 "$OUTPUT_LOG"
    exit 1
fi

echo ""
echo "================================================"
echo "  Simulation Results"
echo "================================================"
echo ""

# Check if RDF file was created
if [ -f "$RDF_OUTPUT" ]; then
    echo "✅ RDF file created: $RDF_OUTPUT"
    num_lines=$(wc -l < "$RDF_OUTPUT")
    echo "   Lines: $num_lines"
else
    echo "⚠️  RDF file not found: $RDF_OUTPUT"
fi

echo ""

# Check if trajectory file was created
if [ -f "dump.ga.lj.lammpstrj" ]; then
    echo "✅ Trajectory file: dump.ga.lj.lammpstrj"
    num_frames=$(grep "ITEM: TIMESTEP" dump.ga.lj.lammpstrj | wc -l)
    echo "   Frames: $num_frames"
else
    echo "⚠️  Trajectory file not found"
fi

echo ""

# Check if final structure was written
if [ -f "outputs/final_structure_ga_lj.data" ]; then
    echo "✅ Final structure: outputs/final_structure_ga_lj.data"
fi

echo ""
echo "================================================"
echo "  Next Step: Analyze S(Q)"
echo "================================================"
echo ""
echo "Run the analysis script:"
echo "  python scripts/run_ga_lj_analyze_sq.py"
echo ""
echo "This will generate:"
echo "  - outputs/ga_lj_sq.png (g(r) and S(Q) plots)"
echo "  - outputs/ga_lj_sq.txt (S(Q) data)"
echo ""
