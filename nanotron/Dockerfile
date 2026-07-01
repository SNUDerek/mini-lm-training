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
    ninja-build \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/huggingface/nanotron.git /workspace/nanotron

WORKDIR /workspace/nanotron

RUN uv venv .venv --python 3.11

RUN uv pip install --upgrade pip setuptools wheel packaging

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install -e .

RUN uv pip install \
    datasets \
    transformers \
    "datatrove[io]" \
    numba \
    wandb

RUN uv pip install \
    ninja \
    triton \
    setuptools \
    wheel \
    packaging

RUN uv pip install \
    "flash-attn>=2.5.0" \
    --no-build-isolation

CMD ["/bin/bash"]