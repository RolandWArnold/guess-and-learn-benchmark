# ────────────────────────────────────────────────────────────────────
# CUDA-runtime image + Python 3.11
# ────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG PY_VER=3.11
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ----- system packages + pip ---------------------------------------
RUN apt-get update && \
    apt-get install -y python${PY_VER} python${PY_VER}-venv python3-pip git && \
    ln -sf /usr/bin/python${PY_VER} /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip

# ----- core Python deps --------------------------------------------
#  * Torch & vision pinned to GPU build (CUDA 12.1 wheel URL)
#  * cloudpickle/boto3/awscli for S3 uploading
RUN pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
      transformers datasets scikit-learn tqdm matplotlib cloudpickle \
      boto3 awscli

# ----- copy your repo ----------------------------------------------
WORKDIR /workspace
COPY . /workspace

# (optional) editable install if you turn this repo into a proper pkg
# RUN pip install --no-cache-dir -e .

# ----- offline-cache defaults --------------------------------------
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# You can override CMD / args from docker-compose
ENTRYPOINT ["python", "-m", "guess_and_learn.run_all"]
