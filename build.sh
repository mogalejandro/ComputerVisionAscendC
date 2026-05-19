#!/usr/bin/env bash
set -e

path_cur=$(dirname "$0")
cd "$path_cur"

# ── Environment ───────────────────────────────────────────────────────────────
# Load CANN environment
if [ -f /usr/local/Ascend/cann-8.5.0/set_env.sh ]; then
    . /usr/local/Ascend/cann-8.5.0/set_env.sh
else
    echo "ERROR: CANN not found at /usr/local/Ascend/cann-8.5.0/"
    exit 1
fi

# Load MindX SDK environment
if [ -f /usr/local/Ascend/mxVision-7.3.0/set_env.sh ]; then
    . /usr/local/Ascend/mxVision-7.3.0/set_env.sh
else
    echo "WARNING: MindX SDK set_env.sh not found, using manual path"
    export MX_SDK_HOME=/usr/local/Ascend/mxVision-7.3.0
fi

export TE_PARALLEL_COMPILER=1

soc="Ascend"
chip_version=$(npu-smi info 2>/dev/null | awk '{print $3}' | grep -m 1 910 || echo "910B")
echo "SoC: ${soc}${chip_version}"

# ── Build ─────────────────────────────────────────────────────────────────────
rm -rf build
mkdir -p build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)" || {
    echo "Build failed"
    exit 1
}

cp infer ../

echo ""
echo "Done — binary: $(dirname "$0")/infer"
