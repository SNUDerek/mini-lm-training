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
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN git clone https://github.com/huggingface/nanotron.git /workspace/nanotron
COPY patches/nanotron-optional-grouped-gemm.patch /tmp/nanotron-optional-grouped-gemm.patch
RUN git -C /workspace/nanotron apply /tmp/nanotron-optional-grouped-gemm.patch

WORKDIR /workspace/nanotron

RUN uv venv .venv --python 3.10

RUN uv pip install --upgrade pip setuptools wheel packaging

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cu124

RUN uv pip install -e .

RUN uv pip install \
    "numpy<2" \
    datasets \
    transformers \
    "datatrove[io,processing]" \
    safetensors \
    dacite \
    torchtyping \
    numba \
    psutil \
    wandb

RUN uv pip install \
    ninja \
    triton \
    setuptools \
    wheel

# # only used for llama, and causing issues, so don't compile, just install pip
RUN uv pip install flash-attn==2.8.3.post1   --no-build-isolation
# RUN MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE \
#     uv pip install \
#     "flash-attn==2.6.3" \  
#     --no-build-isolation \
#     --no-binary flash-attn

CMD ["/bin/bash"]
