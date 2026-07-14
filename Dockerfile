FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    CUDA_HOME=/usr/local/cuda \
    PATH="/root/.local/bin:/workspace/nanotron/.venv/bin:${PATH}"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace/training

RUN uv venv .venv --python 3.12

RUN uv pip install --upgrade pip setuptools wheel packaging

RUN uv pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install \
    datasets \
    transformers>=5.0.0 \
    accelerate>=1.1.0 \
    tensorboardx \
    safetensors \
    torchtyping \
    numba \
    wandb

CMD ["/bin/bash"]
