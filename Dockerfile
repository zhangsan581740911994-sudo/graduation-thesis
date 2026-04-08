FROM python:3.9-slim

# Allow passing proxy values at build time so apt/pip can use them
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV ALL_PROXY=${ALL_PROXY}

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (default China mirror for reliable apt)
ARG USE_CHINA_MIRROR=1
RUN set -e; \
    if [ "$USE_CHINA_MIRROR" = "1" ]; then \
      echo "Using China mirror for apt"; \
      echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ trixie main contrib non-free" > /etc/apt/sources.list; \
      echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian/ trixie-updates main contrib non-free" >> /etc/apt/sources.list; \
      echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian-security/ trixie-security main contrib non-free" >> /etc/apt/sources.list; \
    fi; \
    if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then \
      echo "Acquire::http::Proxy \"${HTTP_PROXY}\";" > /etc/apt/apt.conf.d/95proxy; \
      echo "Acquire::https::Proxy \"${HTTPS_PROXY}\";" >> /etc/apt/apt.conf.d/95proxy; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git wget ca-certificates pkg-config \
      libgl1-mesa-dev libosmesa6-dev patchelf swig ffmpeg \
      libopenblas-dev libopenmpi-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel cython \
  && pip install --no-cache-dir -r /app/requirements.txt

# If pip installed the 'mujoco' package, copy its shared lib into a MUJOCO-like layout
RUN python - <<'PY'
import site, os, shutil
sp = site.getsitepackages()[0]
mj = os.path.join(sp, 'mujoco')
if os.path.isdir(mj):
    os.makedirs('/root/.mujoco/mujoco210/bin', exist_ok=True)
    src = os.path.join(mj, 'libmujoco.so.3.4.0')
    if os.path.exists(src):
        dst = '/root/.mujoco/mujoco210/bin/libmujoco.so'
        shutil.copy(src, dst)
        print('mujoco lib copied to', dst)
    else:
        print('mujoco package found but libmujoco.so.3.4.0 missing:', mj)
else:
    print('mujoco package not found at', mj)
PY

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}

# Copy project files
COPY . /app
RUN pip install --no-cache-dir -e .

# Default to an interactive shell so you can run tests/commands inside the container.
CMD ["/bin/bash"]

