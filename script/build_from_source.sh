#!/bin/bash

# 设置 MKL 路径
MKL_ROOT="/home/wangzehao/intel/oneapi/mkl/2025.2"
export MKLROOT="${MKL_ROOT}"

# 设置库路径以便运行时能找到 MKL
export LD_LIBRARY_PATH="${MKL_ROOT}/lib/intel64:${LD_LIBRARY_PATH}"

# 可选：指定 MKL vendor（如果需要）
# -DBLA_VENDOR=Intel10_64lp  # 使用 lp64 接口
# -DBLA_VENDOR=Intel10_64_dyn  # 使用动态库 libmkl_rt.so
if [ -d "build" ]; then
  rm -rf build
fi

cmake -B build . \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_CUVS=OFF \
  -DFAISS_ENABLE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_C_API=ON \
  -DFAISS_OPT_LEVEL=avx512 \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DFAISS_ENABLE_PYTHON=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -DBUILD_TESTING=OFF \
  -DFAISS_ENABLE_MKL=ON
