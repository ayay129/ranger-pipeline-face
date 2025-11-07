import torch
import torchvision.transforms as transforms
from PIL import Image
import timm  # FairFace backbone = ResNet34 (timm 可用)

# 定义预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# 载入模型
model = torch.load("fairface_alldata_7race.pt", map_location="cpu")
model.eval()

# 读取图片
img = Image.open("/home/rangers/face_workspace/face_insight/facetest.jpeg").convert("RGB")
x = transform(img).unsqueeze(0)

# 推理
with torch.no_grad():
    outputs = model(x)
    probs = torch.nn.Softmax(dim=1)(outputs)

print("分类概率:", probs)

