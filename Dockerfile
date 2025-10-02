# GPU-ready image for GLORY/GLoCIM (PyTorch 2.5.1 + CUDA 12.1)
# Builds a runnable container that executes `python src/main.py ...`.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PROJECT_ROOT=/app \
    WANDB_MODE=offline

WORKDIR /app

# System deps + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-distutils \
    ca-certificates git curl wget unzip \
 && rm -rf /var/lib/apt/lists/*

# Create venv and upgrade pip
RUN python3.10 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip setuptools wheel

# Copy minimal files first to leverage Docker layer caching
COPY requirements.txt ./

# Install core torch stack (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyG family matching torch 2.5.1 cu121
RUN pip install --no-cache-dir \
    torch-geometric==2.6.1 \
    torch-scatter==2.1.2+pt25cu121 \
    torch-sparse==0.6.18+pt25cu121 \
    torch-cluster==1.6.3+pt25cu121 \
    torch-spline-conv==1.2.2+pt25cu121 \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Install the rest of Python deps from requirements.txt, excluding torch* lines
RUN python - <<'PY'
import re
src = open('requirements.txt').read().splitlines()
keep = [l for l in src if not re.match(r'^(torch|torchvision|torchaudio|torch[-_].*)\b', l.strip(), flags=re.I)]
open('requirements.notorch.txt','w').write('\n'.join(keep) + '\n')
PY
RUN pip install --no-cache-dir -r requirements.notorch.txt

# Copy project sources
COPY . /app

# Create standard dirs (mounts may override)
RUN mkdir -p /app/data /app/logs /app/checkpoint /app/outputs

# Default command can be overridden at `docker run ... -- <args>`
ENTRYPOINT ["python", "src/main.py"]
CMD ["model=GLORY", "dataset=MINDsmall", "reprocess=True"]

