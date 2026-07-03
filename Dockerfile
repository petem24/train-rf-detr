FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG RFDETR_VERSION=1.8.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATASET_DIR=/workspace/dataset \
    OUTPUT_DIR=/workspace/output \
    DATASET_FORMAT=coco \
    MODEL_TASK=detection \
    MODEL_VARIANT=medium \
    EPOCHS=100 \
    BATCH_SIZE=8 \
    GRAD_ACCUM_STEPS=2 \
    LR=1e-4 \
    EXTRA_TRAIN_ARGS_JSON='{"eval_interval":5,"log_per_class_metrics":false,"pin_memory":true,"persistent_workers":true,"prefetch_factor":4}'

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libgl1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "rfdetr[train,loggers,onnx]==${RFDETR_VERSION}" pyyaml

COPY train_rfdetr.py /opt/rfdetr/train_rfdetr.py

CMD ["python", "/opt/rfdetr/train_rfdetr.py"]
