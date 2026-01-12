#!/bin/bash
# Bimodal Grid Search V3
# σ12: 1.10x, 1.15x, 1.20x
# Ga1: 35%, 50%, 65%

cd /home/yuki/lammps_settings_obata/hirataken20251122-2

echo "=== Bimodal Grid Search V3 ==="
echo "Start time: $(date)"
echo ""

# List of input files
INPUT_FILES=(
    "in.sig12_110_ga1_35"
    "in.sig12_110_ga1_50"
    "in.sig12_110_ga1_65"
    "in.sig12_115_ga1_35"
    "in.sig12_115_ga1_50"
    "in.sig12_115_ga1_65"
    "in.sig12_120_ga1_35"
    "in.sig12_120_ga1_50"
    "in.sig12_120_ga1_65"
)

TOTAL=${#INPUT_FILES[@]}
COUNT=0

for infile in "${INPUT_FILES[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Running: $infile"
    echo "  Started: $(date '+%H:%M:%S')"

    # Run LAMMPS with Kokkos GPU
    lmp -k on g 1 -sf kk -in outputs/bimodal_search_v3/$infile \
        > outputs/bimodal_search_v3/log.${infile#in.} 2>&1

    if [ $? -eq 0 ]; then
        echo "  Completed successfully"
    else
        echo "  ERROR: Simulation failed!"
    fi
    echo ""
done

echo "=== All simulations completed ==="
echo "End time: $(date)"
