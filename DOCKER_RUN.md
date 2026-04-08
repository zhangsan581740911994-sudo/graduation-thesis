# EDP 使用 Docker 运行说明（推荐，Python 3.9）

项目已配置为在 **Docker** 内使用 **Python 3.9** 运行，避免本机 Python 3.11+ 与 mujoco_py/d4rl 的兼容问题。

## 1. 构建镜像

在项目根目录执行（首次或修改 Dockerfile/requirements 后）：

```bash
docker build -t edp .
```

国内可加速 apt/pip（可选）：

```bash
docker build --build-arg USE_CHINA_MIRROR=1 -t edp .
```

## 2. 小规模验证（约几分钟）

确认依赖和流程正常：

```bash
docker run --rm -it edp /bin/bash -c "bash /app/scripts/run_small.sh"
```

## 3. 正式训练示例

例如 walker2d-medium-v2，TD3：

```bash
docker run --rm -it edp python -m diffusion.trainer \
  --env=walker2d-medium-v2 \
  --logging.output_dir=./experiment_output \
  --algo_cfg.loss_type=TD3
```

如需把结果保留到本机，可挂载目录：

```bash
docker run --rm -it -v "%cd%\experiment_output:/app/experiment_output" edp python -m diffusion.trainer --env=walker2d-medium-v2 --logging.output_dir=./experiment_output --algo_cfg.loss_type=TD3
```

## 4. 进入容器调试

```bash
docker run --rm -it edp /bin/bash
# 在容器内：
pip list
python -c "import d4rl; print('d4rl ok')"
python -m diffusion.trainer --help
```

## 5. WSL 下使用

若在 WSL 里用 Docker，路径用 Linux 形式，例如挂载：

```bash
docker run --rm -it -v "$(pwd)/experiment_output:/app/experiment_output" edp bash /app/scripts/run_small.sh
```
