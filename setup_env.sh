#!/bin/bash
set -e

# setup_env.sh - Automated dependency installer for FHE-RaiderSTREAM
# Installs everything into the ./env folder

USE_HEXL=false

# Simple argument parsing
for arg in "$@"; do
    case $arg in
        --hexl)
            USE_HEXL=true
            shift
            ;;
    esac
done

PROJECT_ROOT=$(pwd)
ENV_DIR="$PROJECT_ROOT/env"
SRC_DIR="$ENV_DIR/src"
BUILD_DIR="$ENV_DIR/build"
INSTALL_DIR="$ENV_DIR/dist"

mkdir -p "$SRC_DIR" "$BUILD_DIR" "$INSTALL_DIR"

echo "==> Target installation directory: $INSTALL_DIR"
echo "==> HEXL acceleration: $USE_HEXL"

# Helper for environment variables during build
export PATH="$INSTALL_DIR/bin:$PATH"
export CPATH="$INSTALL_DIR/include:$CPATH"
export LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:$INSTALL_DIR/lib64:$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_DIR/lib/pkgconfig:$INSTALL_DIR/lib64/pkgconfig:$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="$INSTALL_DIR:$CMAKE_PREFIX_PATH"
export CC=$(which mpicc)
export CXX=$(which mpicxx)

# 1. Cereal (Header-only)
if [ ! -d "$INSTALL_DIR/include/cereal" ]; then
    echo "==> Installing Cereal..."
    cd "$SRC_DIR"
    if [ ! -d "cereal" ]; then
        git clone https://github.com/USCiLab/cereal.git
    fi
    mkdir -p "$INSTALL_DIR/include"
    cp -r cereal/include/cereal "$INSTALL_DIR/include/"
else
    echo "--> Cereal already installed."
fi

# 2. Google Benchmark
if [ ! -f "$INSTALL_DIR/lib/libbenchmark.a" ] && [ ! -f "$INSTALL_DIR/lib64/libbenchmark.a" ]; then
    echo "==> Building Google Benchmark..."
    cd "$SRC_DIR"
    if [ ! -d "benchmark" ]; then
        git clone https://github.com/google/benchmark.git
    fi
    mkdir -p "$BUILD_DIR/benchmark"
    cd "$BUILD_DIR/benchmark"
    cmake "$SRC_DIR/benchmark" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DBENCHMARK_ENABLE_TESTING=OFF \
          -DBENCHMARK_ENABLE_GTEST_TESTS=OFF \
          -DBENCHMARK_USE_BUNDLED_GTEST=OFF \
          -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    make install
else
    echo "--> Google Benchmark already installed."
fi

# 3. Intel HEXL (Optional)
if [ "$USE_HEXL" = true ]; then
    if [ ! -f "$INSTALL_DIR/lib/libhexl.a" ] && [ ! -f "$INSTALL_DIR/lib/libhexl.so" ]; then
        echo "==> Building Intel HEXL v1.2.6..."
        cd "$SRC_DIR"
        if [ ! -d "hexl" ]; then
            git clone --branch v1.2.6 https://github.com/intel/hexl.git
        fi
        mkdir -p "$BUILD_DIR/hexl"
        cd "$BUILD_DIR/hexl"
        cmake "$SRC_DIR/hexl" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DHEXL_BENCHMARK=OFF \
              -DHEXL_COVERAGE=OFF \
              -DHEXL_DOCS=OFF \
              -DHEXL_EXPERIMENTAL=OFF \
              -DHEXL_TESTING=OFF \
              -DHEXL_SHARED_LIB=OFF \
              -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
        make install
    else
        echo "--> Intel HEXL already installed."
    fi
fi

# 4. OpenFHE
if [ ! -d "$INSTALL_DIR/include/openfhe" ]; then
    VERSION="v1.5.0"
    echo "==> Building OpenFHE $VERSION..."
    cd "$SRC_DIR"
    
    # 4.1. Fetch OpenFHE development base
    if [ ! -d "openfhe-development" ]; then
        git clone --branch $VERSION https://github.com/openfheorg/openfhe-development.git
    fi
    
    if [ "$USE_HEXL" = true ]; then
        # 4.2. Fetch OpenFHE-HEXL overlay
        if [ ! -d "openfhe-hexl" ]; then
            git clone --branch v1.5.0.0 https://github.com/openfheorg/openfhe-hexl.git
        fi
        
        # 4.3. Apply HEXL Overlay
        echo "==> Applying OpenFHE-HEXL overlay..."
        cp -rv openfhe-hexl/benchmark openfhe-development/
        cp -rv openfhe-hexl/configure openfhe-development/
        cp -rv openfhe-hexl/src openfhe-development/
        cp -rv openfhe-hexl/third-party openfhe-development/
        cp -v openfhe-hexl/CMakeLists.txt openfhe-development/
        cp -v openfhe-hexl/CMakeLists.User.txt openfhe-development/
        cp -v openfhe-hexl/OpenFHEConfig.cmake.in openfhe-development/
        
        HEXL_FLAGS="-DWITH_INTEL_HEXL=ON -DINTEL_HEXL_PREBUILT=ON -DINTEL_HEXL_HINT_DIR=$INSTALL_DIR"
    else
        HEXL_FLAGS="-DWITH_INTEL_HEXL=OFF"
    fi
    
    # 4.4. Fix for cereal paths (similar to Nix postPatch)
    mkdir -p "$SRC_DIR/openfhe-development/third-party/cereal"
    cp -r "$INSTALL_DIR/include/cereal" "$SRC_DIR/openfhe-development/third-party/cereal/include"

    # 4.5. Build OpenFHE
    mkdir -p "$BUILD_DIR/openfhe"
    cd "$BUILD_DIR/openfhe"
    cmake "$SRC_DIR/openfhe-development" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DGIT_SUBMOD_AUTO=OFF \
          $HEXL_FLAGS \
          -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
          -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    make install
else
    echo "--> OpenFHE already installed."
fi

# 5. Generate Activation Script
cat <<EOF > "$ENV_DIR/activate"
# Source this script to set up the environment: source env/activate

export ENV_DIR="$INSTALL_DIR"
export PATH="\$ENV_DIR/bin:\$PATH"
export CPATH="\$ENV_DIR/include:\$CPATH"
export LIBRARY_PATH="\$ENV_DIR/lib:\$ENV_DIR/lib64:\$LIBRARY_PATH"
export LD_LIBRARY_PATH="\$ENV_DIR/lib:\$ENV_DIR/lib64:\$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="\$ENV_DIR/lib:\$ENV_DIR/lib64:\$DYLD_LIBRARY_PATH"
export PKG_CONFIG_PATH="\$ENV_DIR/lib/pkgconfig:\$ENV_DIR/lib64/pkgconfig:\$PKG_CONFIG_PATH"
export CMAKE_PREFIX_PATH="\$ENV_DIR:\$CMAKE_PREFIX_PATH"

echo "FHE-RaiderSTREAM environment activated (./env/dist)"
EOF

chmod +x "$ENV_DIR/activate"

echo ""
echo "==> Setup Complete!"
echo "==> To activate the environment, run: source env/activate"
echo "==> Then build the project: ./build.sh"
