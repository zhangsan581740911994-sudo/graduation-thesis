#!/usr/bin/env bash
# AutoDL / Linux：在跑 diffusion 训练前 source 本文件，减少 JAX/XLA 的 ptxas、动态库问题。
# 用法（项目根目录、已 activate venv）：
#   source scripts/source_autodl_jax_gpu_env.sh
#   SEEDS="42 43 44" bash scripts/run_ablation_multimodal_csi1000.sh
#
# 若仍报 ptxas / sm_90a：先
#   pip install 'nvidia-cuda-nvcc-cu12>=12.2'
# 再重新 source 本文件。

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "warning: VIRTUAL_ENV 未设置，请先 source venv/bin/activate" >&2
fi

# pip 安装的 CUDA nvcc/ptxas（与 jaxlib+cuda12 常更匹配；需: pip install nvidia-cuda-nvcc-cu12）
if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
  _NVCC_BIN="$(python - <<'PY' 2>/dev/null || true
import pathlib
try:
    import nvidia.cuda_nvcc
    p = pathlib.Path(nvidia.cuda_nvcc.__file__).resolve().parent / "bin"
    if p.is_dir():
        print(p)
except Exception:
    pass
PY
)"
  if [[ -n "${_NVCC_BIN}" && -d "${_NVCC_BIN}" ]]; then
    export PATH="${_NVCC_BIN}:${PATH}"
  fi
  # 与训练前说明一致：让运行时能找到 cuSPARSE/cuDNN 等
  export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cublas/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cusolver/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cufft/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib:${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

# 系统镜像里的 CUDA（作补充；优先使用较新的 ptxas）
for _cuda in /usr/local/cuda-12 /usr/local/cuda-12.4 /usr/local/cuda-12.2 /usr/local/cuda; do
  if [[ -d "${_cuda}/bin" ]]; then
    export PATH="${_cuda}/bin:${PATH}"
    export CUDA_HOME="${_cuda}"
    break
  fi
done

if [[ -n "${CUDA_HOME:-}" ]]; then
  export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CUDA_HOME}"
fi

echo "PATH 前部 ptxas: $(command -v ptxas 2>/dev/null || echo '(未找到)')"
if command -v ptxas >/dev/null 2>&1; then
  ptxas --version 2>&1 | head -1 || true
fi
