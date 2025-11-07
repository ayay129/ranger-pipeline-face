FROM cuda-python:v3.10
COPY ./code /app
COPY ./models /root/.insightface/models
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && \
    apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 
# 升级 pip
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir insightface onnxruntime-gpu==1.19.0 fastapi uvicorn opencv-python python-multipart && \
    pip install --no-cache-dir 'numpy<2'
ENV PYTHONUNBUFFERED=1

# 暴露端口8000
EXPOSE 8000

# 启动命令
CMD ["python", "face_fastapi.py", "--host", "0.0.0.0", "--port", "8000"]
