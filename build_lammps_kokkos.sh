#!/bin/bash
# Script to build LAMMPS with KOKKOS support for H100

set -e

echo "================================================"
echo "  Building LAMMPS with KOKKOS for H100"
echo "================================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ Error: nvcc not found. Please install CUDA Toolkit first:"
    echo "   wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run"
    echo "   sudo sh cuda_12.3.0_545.23.06_linux.run"
    exit 1
fi

echo "✅ CUDA found: $(nvcc --version | grep release)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
sudo apt update
sudo apt install -y cmake build-essential git

# Check if LAMMPS source exists
LAMMPS_DIR="$HOME/lammps"
if [ -d "$LAMMPS_DIR" ]; then
    echo "📁 LAMMPS source found at $LAMMPS_DIR"
else
    echo "📥 Downloading LAMMPS source..."
    cd ~
    git clone -b stable --depth 1 https://github.com/lammps/lammps.git
    echo "✅ LAMMPS downloaded"
fi

cd "$LAMMPS_DIR"

# Clean previous build
if [ -d "build" ]; then
    echo "🧹 Cleaning previous build..."
    rm -rf build
fi

mkdir build
cd build

echo ""
echo "🔨 Configuring LAMMPS with KOKKOS for H100..."
echo ""

# Configure with KOKKOS for H100 (Hopper architecture)
cmake ../cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DPKG_KOKKOS=yes \
  -DPKG_MANYBODY=yes \
  -DKokkos_ENABLE_CUDA=yes \
  -DKokkos_ARCH_HOPPER90=yes \
  -DCMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local

if [ $? -ne 0 ]; then
    echo "❌ Configuration failed. Trying alternative method..."
    
    # Alternative: use GPU package instead
    cmake ../cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DPKG_GPU=yes \
      -DPKG_MANYBODY=yes \
      -DGPU_API=cuda \
      -DGPU_ARCH=sm_90 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local
fi

echo ""
echo "🔨 Building LAMMPS (this may take 10-20 minutes)..."
echo ""

# Build with all available cores
make -j $(nproc)

echo ""
echo "📦 Installing LAMMPS..."
make install

# Add to PATH if not already there
if ! grep -q "/.local/bin" ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "✅ Added LAMMPS to PATH in ~/.bashrc"
fi

echo ""
echo "================================================"
echo "  ✅ LAMMPS Build Complete!"
echo "================================================"
echo ""
echo "To use the new LAMMPS:"
echo "  1. Close and reopen your terminal, OR"
echo "  2. Run: source ~/.bashrc"
echo ""
echo "Then test with:"
echo "  lmp -help | grep KOKKOS"
echo ""
echo "To run simulation:"
echo "  lmp -k on g 1 -sf kk -in in.ga_H100"
echo ""
