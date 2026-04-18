#!/usr/bin/env bash
# AutoDL / Linux x86_64 / Python 3.12：安装带 CUDA 的 JAX（与 Flax 0.8.x 兼容）
#
# 背景：jax_cuda_releases 对 cp312 的「一体式」GPU jaxlib 只提供到 0.4.29+cuda12.cudnn91，
#       没有 0.4.35+cuda12；若 pip 只装 jax==0.4.35，jaxlib 常为 CPU，或误装 jax-cuda12-plugin 导致冲突。
# 用法：在项目根目录
#   source venv/bin/activate
#   bash scripts/install_jax_gpu_autodl_cp312.sh
# 然后设置 LD_LIBRARY_PATH（见 docs/thesis_materials/CSI1000_Top50实验步骤.md）
set -euo pipefail

JAX_VER="0.4.29"
JAXLIB_TAG="0.4.29+cuda12.cudnn91"
GCS="-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"

echo ">>> 卸载可能冲突的 jax 相关包"
pip uninstall -y jax jaxlib jax-cuda12-plugin jax-cuda11-plugin jax-cuda-plugin 2>/dev/null || true

echo ">>> 安装 jax==${JAX_VER} + jaxlib==${JAXLIB_TAG}（CUDA12，cp312 x86_64）"
pip install "jax==${JAX_VER}" "jaxlib==${JAXLIB_TAG}" ${GCS}

echo ">>> 固定训练栈（用约束文件防止 pip 把 jax 升到 0.10）"
TMP_CON="$(mktemp)"
echo "jax==${JAX_VER}" > "${TMP_CON}"
pip install -c "${TMP_CON}" flax==0.8.4 optax==0.1.9 orbax-checkpoint==0.5.16 chex==0.1.86
rm -f "${TMP_CON}"

echo ">>> 验证（应出现 GpuDevice / CudaDevice，而非仅 CpuDevice）"
python -c "import jax; print('jax', jax.__version__); print('devices:', jax.devices()); print('default backend:', jax.default_backend())"
