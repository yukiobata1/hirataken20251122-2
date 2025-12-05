#!/bin/bash
# H100 GPU-Accelerated LAMMPS Execution Script for Pure Ga LJ

set -e

echo "================================================"
echo "  H100 GPU-Accelerated Gallium LJ Simulation"
echo "================================================"
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ Error: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

echo "🔍 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check for LAMMPS with KOKKOS
if ! lmp -help 2>/dev/null | grep -q "KOKKOS"; then
    echo "❌ Error: LAMMPS not compiled with KOKKOS support"
    echo "Please build LAMMPS with KOKKOS package"
    exit 1
fi

echo "✅ LAMMPS with KOKKOS found"
echo ""

# Check for input file
INPUT_FILE=${1:-"inputs/in.ga_lj_150C"} # Default to our new input file
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "📄 Using input file: $INPUT_FILE"
echo ""

# Set GPU
export CUDA_VISIBLE_DEVICES=0
echo "🎮 Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Start GPU monitoring in background
echo "📊 Starting GPU monitor (nvidia-smi in background)..."
watch -n 2 nvidia-smi > gpu_monitor.log 2>&1 &
MONITOR_PID=$!

# Execution
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="outputs/output_ga_lj_${TIMESTAMP}.log"

echo "🚀 Starting simulation..."
echo "   Log file: $OUTPUT_LOG"
echo ""

# Ensure outputs directory exists
mkdir -p outputs

# Run LAMMPS with KOKKOS
# The -k on g 1 option specifies KOKKOS package, with 1 GPU.
# -sf kk option tells LAMMPS to append /kk to force styles that support it.
time lmp -k on g 1 -sf kk -in "$INPUT_FILE" > "$OUTPUT_LOG" 2>&1

EXIT_CODE=$?

# Stop GPU monitor
kill $MONITOR_PID 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Simulation completed successfully!"
    echo ""
    
    # Extract performance info (if present)
    echo "📈 Performance Summary:"
    grep "Loop time of " "$OUTPUT_LOG" | tail -1 || echo "No performance summary found."
    echo ""
    
    # Extract final results
    echo "📊 Final Results:"
    # This grep needs to match the print statements in the LAMMPS input script
    grep -A 5 "FINAL RESULTS:" "$OUTPUT_LOG" || echo "No final results found in log."
    echo ""
    
    # Check output files
    echo "📁 Generated files:"
    ls -lh rdf.dat outputs/final_structure_ga_lj.data dump.ga.lj.lammpstrj 2>/dev/null | awk '{print "   "$9, "-", $5}' || echo "Output files not found or empty."
    echo ""
    
else
    echo "❌ Simulation failed with exit code $EXIT_CODE"
    echo "   Check log file: $OUTPUT_LOG"
    exit $EXIT_CODE
fi

echo "================================================"
echo "  Simulation Complete!"
echo "================================================"
