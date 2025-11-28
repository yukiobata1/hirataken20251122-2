#!/bin/bash
# Quick test script to verify H100 GPU setup for EPSR

set -e

PROJECT_DIR="/home/yuki/lammps_settings_obata/hirataken20251122-2"
cd "$PROJECT_DIR"

echo "================================================"
echo "  H100 GPU Setup Verification"
echo "================================================"
echo ""

# Test 1: NVIDIA GPU
echo "✓ Test 1: Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    echo "  GPU: $GPU_NAME"
    echo "  Memory: ${GPU_MEM} MB"
    echo "  Driver: $DRIVER"

    if echo "$GPU_NAME" | grep -qi "H100"; then
        echo "  ✅ H100 detected!"
    else
        echo "  ⚠️  Warning: GPU is not H100 ($GPU_NAME)"
        echo "     Script will still work but may be slower"
    fi
else
    echo "  ❌ FAILED: nvidia-smi not found"
    echo "     Please install NVIDIA drivers"
    exit 1
fi
echo ""

# Test 2: CUDA
echo "✓ Test 2: Checking CUDA..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    echo "  CUDA: $CUDA_VERSION"
    echo "  ✅ CUDA Toolkit found"
else
    echo "  ⚠️  Warning: nvcc not found"
    echo "     CUDA Toolkit may not be installed"
    echo "     LAMMPS may still work if already built"
fi
echo ""

# Test 3: LAMMPS
echo "✓ Test 3: Checking LAMMPS..."
if command -v lmp &> /dev/null; then
    LMP_PATH=$(which lmp)
    echo "  Path: $LMP_PATH"
    echo "  ✅ LAMMPS found"
else
    echo "  ❌ FAILED: lmp command not found"
    echo ""
    echo "To install LAMMPS with Kokkos:"
    echo "  bash scripts/build_lammps_kokkos.sh"
    exit 1
fi
echo ""

# Test 4: LAMMPS Kokkos Support
echo "✓ Test 4: Checking LAMMPS Kokkos support..."
if lmp -help 2>&1 | grep -q "KOKKOS"; then
    echo "  ✅ KOKKOS package found"

    # Show enabled packages
    echo "  Packages:"
    lmp -help 2>&1 | grep -A 20 "Installed packages:" | head -15 | sed 's/^/    /'
else
    echo "  ❌ FAILED: KOKKOS package not found"
    echo ""
    echo "Your LAMMPS was not built with Kokkos support."
    echo "Please rebuild:"
    echo "  bash scripts/build_lammps_kokkos.sh"
    exit 1
fi
echo ""

# Test 5: Project Files
echo "✓ Test 5: Checking project files..."

FILES_OK=true

if [ ! -f "inputs/initial_structure.data" ]; then
    echo "  ⚠️  Missing: inputs/initial_structure.data"
    echo "     Run: python3 scripts/create_initial_structure.py"
    FILES_OK=false
fi

if [ ! -f "data/g_exp.dat" ]; then
    echo "  ⚠️  Missing: data/g_exp.dat"
    echo "     Experimental data needs to be provided"
    FILES_OK=false
fi

if [ ! -f "data/ep_GaGa.table" ]; then
    echo "  ⚠️  Missing: data/ep_*.table"
    echo "     Run: python3 scripts/update_ep.py"
    FILES_OK=false
fi

if [ ! -f "inputs/in.egain_lj_H100" ]; then
    echo "  ❌ Missing: inputs/in.egain_lj_H100"
    FILES_OK=false
fi

if [ ! -f "inputs/in.egain_epsr_H100" ]; then
    echo "  ❌ Missing: inputs/in.egain_epsr_H100"
    FILES_OK=false
fi

if [ "$FILES_OK" = true ]; then
    echo "  ✅ All required files present"
else
    echo ""
    echo "Some files are missing. Running setup..."

    if [ ! -f "inputs/initial_structure.data" ]; then
        echo "Creating initial structure..."
        python3 scripts/create_initial_structure.py
    fi

    if [ ! -f "data/ep_GaGa.table" ]; then
        echo "Initializing empirical potentials..."
        python3 scripts/update_ep.py
    fi

    if [ ! -f "data/g_exp.dat" ]; then
        echo ""
        echo "⚠️  Please provide experimental g(r) data:"
        echo "   - Save to: data/g_exp.dat"
        echo "   - Format: 2 columns (r in Å, g(r))"
    fi
fi
echo ""

# Test 6: Python Dependencies
echo "✓ Test 6: Checking Python dependencies..."
PYTHON_OK=true

for pkg in numpy scipy matplotlib; do
    if python3 -c "import $pkg" 2>/dev/null; then
        VERSION=$(python3 -c "import $pkg; print($pkg.__version__)")
        echo "  ✅ $pkg ($VERSION)"
    else
        echo "  ❌ Missing: $pkg"
        PYTHON_OK=false
    fi
done

if [ "$PYTHON_OK" = false ]; then
    echo ""
    echo "Install missing packages:"
    echo "  pip install numpy scipy matplotlib"
fi
echo ""

# Summary
echo "================================================"
echo "  Summary"
echo "================================================"
echo ""

if [ "$FILES_OK" = true ] && [ "$PYTHON_OK" = true ]; then
    echo "✅ All tests passed! Ready to run EPSR."
    echo ""
    echo "Next steps:"
    echo "  1. Test LJ simulation:"
    echo "     bash scripts/run_epsr_H100.sh --test-lj"
    echo ""
    echo "  2. Run full EPSR:"
    echo "     bash scripts/run_epsr_H100.sh"
    echo ""
else
    echo "⚠️  Some issues found. Please resolve before running EPSR."
    echo ""
fi
