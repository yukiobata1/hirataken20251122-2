#!/bin/bash
# H100 GPU-Accelerated LAMMPS Execution Script

set -e

echo "================================================"
echo "  H100 GPU-Accelerated Gallium Simulation"
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
INPUT_FILE=${1:-"in.ga_H100"}
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

echo "📄 Using input file: $INPUT_FILE"
echo ""

# Check for EAM potential
if [ ! -f "Ga_belashchenko2012.eam.alloy" ]; then
    echo "❌ Error: EAM potential file not found"
    exit 1
fi

echo "✅ EAM potential found"
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
OUTPUT_LOG="output_h100_${TIMESTAMP}.log"

echo "🚀 Starting simulation..."
echo "   Log file: $OUTPUT_LOG"
echo "   Estimated time: 5-10 minutes"
echo ""

# Run LAMMPS with KOKKOS on H100
time lmp -k on g 1 -sf kk -in "$INPUT_FILE" > "$OUTPUT_LOG" 2>&1

EXIT_CODE=$?

# Stop GPU monitor
kill $MONITOR_PID 2>/dev/null || true

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Simulation completed successfully!"
    echo ""
    
    # Extract performance info
    echo "📈 Performance Summary:"
    grep "Performance:" "$OUTPUT_LOG" | tail -1
    echo ""
    
    # Extract final results
    echo "📊 Final Results:"
    grep -A 5 "GPU SIMULATION RESULTS" "$OUTPUT_LOG" || \
    grep "Density:" "$OUTPUT_LOG" | tail -1
    echo ""
    
    # Check output files
    echo "📁 Generated files:"
    ls -lh dump.ga.gpu.lammpstrj rdf.ga.gpu.dat msd.ga.gpu.dat 2>/dev/null | awk '{print "   "$9, "-", $5}'
    echo ""
    
else
    echo "❌ Simulation failed with exit code $EXIT_CODE"
    echo "   Check log file: $OUTPUT_LOG"
    exit $EXIT_CODE
fi

echo "================================================"
echo "  Simulation Complete!"
echo "================================================"
