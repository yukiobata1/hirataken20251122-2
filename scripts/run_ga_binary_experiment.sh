#!/bin/bash

# Setup script for Ga Binary System Experiment (CPU Version)

echo "Phase 1: Creating binary structure..."
python3 scripts/create_binary_structure.py

echo "Phase 2: Running Simulation Run 0 (Baseline)..."
LMP_CMD="lmp"

$LMP_CMD -in inputs/in.ga_binary_run0 -log outputs/log.binary_run0
echo "Run 0 complete. Check outputs/log.binary_run0"

echo "Phase 2: Running Simulation Run 1 (Size Difference 10%)..."
$LMP_CMD -in inputs/in.ga_binary_run1 -log outputs/log.binary_run1

echo "Simulation complete. Results in outputs/"
echo " - Run 0 RDF: outputs/rdf_binary_run0.dat"
echo " - Run 1 RDF: outputs/rdf_binary_run1.dat"