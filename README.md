# algo-cv-pipeline-face

人脸检测/关键点/特征提取与属性识别（年龄、性别、种族）的推理服务与示例代码。服务采用 FastAPI，模型默认使用 InsightFace 的自动下载与管理（`buffalo_l` 套件）。同时提供一套 ONNX 模型包以便在需要时离线使用或自定义加载。

## 目录结构

```
├── code/                     # 服务与模型调用代码
│   ├── face_fastapi.py       # FastAPI 服务（HTML上传表单 + API）
│   ├── face_model.py         # 使用 InsightFace 的模型封装
│   ├── face_model_cpu.py     # 使用 InsightFace 的CPU/GPU配置示例
│   ├── test_attributes.py    # 属性识别测试脚本
│   └── requirements.txt      # 运行依赖
├── models/
│   └── buffalo_l/            # 本地ONNX模型（若走离线/自定义加载）
├── algo-cv-pipeline-face-buffalo-l/
│   ├── buffalo_l/            # 同步到 Hugging Face 的ONNX模型包（本地副本）
│   └── README.md             # 模型卡（说明与用法）
├── Dockerfile                # 基础镜像构建
├── Dockerfile.cuda.py310     # CUDA/ONNXRuntime GPU 构建（示例）
└── .gitignore                # 忽略大模型文件等
```

## 环境准备

- Python 3.9+（建议）
- 推荐使用虚拟环境或 Conda 环境

安装依赖：

```bash
# 在项目根目录
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# 如使用 InsightFace + GPU：
# pip install onnxruntime-gpu insightface
```

## 运行服务

两种方式均可：

- 直接运行 Python：

```bash
python code/face_fastapi.py --host 0.0.0.0 --port 8112
```

- 使用 uvicorn：

```bash
uvicorn code.face_fastapi:app --host 0.0.0.0 --port 8112
```

启动后访问：
- 浏览器打开 `http://localhost:8112/`（内置一个简易上传表单页面）
- 健康检查：`GET /health`
- 运行指标：`GET /metrics`

## API 使用

- 表单上传并可选请求属性（性别/年龄/种族）：

```bash
curl -X POST \
  -F "file=@/path/to/your/image.jpg" \
  -F "gender=1" -F "age=1" -F "race=1" \
  http://localhost:8112/
```

返回示例（每个检测到的人脸一条记录）：

```json
[
  {
    "bboxes": [x1, y1, x2, y2, score],
    "feature": [ ... 512维/特征 ... ],
    "race": "Asian",
    "gender": "Male",
    "age": "28"
  }
]
```

## 模型管理

### 默认（推荐）：InsightFace 自动下载

代码中的 `FaceModel` 使用 InsightFace 的 `FaceAnalysis(name='buffalo_l')`。首次运行时会自动从官方源下载所需模型到本地缓存，无需手动管理。

- GPU 环境：设置 `providers=['CUDAExecutionProvider']`；确保安装 `onnxruntime-gpu`
- CPU 环境：可改用 `providers=['CPUExecutionProvider']`

### 使用 Hugging Face 存储 ONNX 模型

如果你希望通过 Hugging Face Hub 管理模型包（例如仓库ID：`<你的用户名>/algo-cv-pipeline-face-buffalo-l`），可以在部署时按需下载：

```python
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "<你的用户名>/algo-cv-pipeline-face-buffalo-l"
LOCAL_DIR = Path("models") / "buffalo_l"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

for fname in [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
]:
    hf_hub_download(repo_id=REPO_ID, filename=fname, repo_type="model", local_dir=str(LOCAL_DIR))
```

- 私有仓库请先执行 `hf auth login` 或在环境变量中设置 `HUGGINGFACE_HUB_TOKEN`

### 推送模型到 Hugging Face（CLI）

```bash
# 登录（交互或用令牌）
hf auth login
# 或：hf auth login --token <你的令牌>

# 创建模型仓库（建议带用户名命名空间）
hf repo create <你的用户名>/algo-cv-pipeline-face-buffalo-l --repo-type model

# 上传本地ONNX文件夹（仅ONNX）
hf upload \
  --repo-id <你的用户名>/algo-cv-pipeline-face-buffalo-l \
  --repo-type model \
  models/buffalo_l \
  --include "**/*.onnx" \
  --commit-message "Add buffalo_l ONNX models"
```

## 代码推送到 GitHub

仓库已在 `.gitignore` 中忽略大模型文件；用占位 `models/.gitkeep` 保留目录结构。

```bash
# 初始化并首次推送
git init
mkdir -p models && touch models/.gitkeep

git add code Dockerfile Dockerfile.cuda.py310 .gitignore models/.gitkeep
git commit -m "Initial commit: code, requirements, dockerfiles"

git branch -M main
git remote add origin git@github.com:<你的用户名>/algo-cv-pipeline-face.git
git push -u origin main
```

## Docker（可选）

基础镜像构建（CPU 示例）：

```bash
docker build -t algo-cv-face:cpu -f Dockerfile .
docker run --rm -p 8112:8112 algo-cv-face:cpu
```

CUDA/ONNXRuntime GPU 示例请参考 `Dockerfile.cuda.py310`，并确保宿主机有合适的 NVIDIA 驱动与 Docker GPU 运行时。

## 测试

```bash
python code/test_attributes.py
```

## 注意事项

- 属性识别中的“种族”示例实现包含演示/占位逻辑，仅用于技术流程演示，不适合生产使用。
- InsightFace/ONNXRuntime 的性能与精度依赖具体硬件与提供者配置。
- 如将 ONNX 模型公开分发，请确认上游许可协议并在 Hugging Face 模型卡中注明。

## 致谢

- InsightFace 项目：https://github.com/deepinsight/insightface
- ONNX Runtime：https://onnxruntime.ai/

## 许可

根据你使用的上游模型与代码许可选择合适的开源协议（如 Apache-2.0/MIT 等）；如已有明确许可，请在此处补充，并在 Hugging Face 模型卡中同步。
