FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    python-is-python3 \
    libgl1 \
    libglib2.0-0 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./code /app
COPY ./models /models

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    python3 -m pip show onnxruntime-gpu

ENV PYTHONUNBUFFERED=1
ENV FACE_ONNX_DIR=/models/buffalo_l
ENV ORT_GRAPH_OPT_LEVEL=ORT_ENABLE_BASIC
ENV ORT_CUDNN_CONV_ALGO_SEARCH=DEFAULT
ENV ORT_CUDNN_CONV_USE_MAX_WORKSPACE=0

# 暴露端口8000
EXPOSE 8000

# 启动命令
CMD ["python3", "face_fastapi.py", "--host", "0.0.0.0", "--port", "8000"]
