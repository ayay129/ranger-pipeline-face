import insightface
from insightface.app import FaceAnalysis

# 创建分析器
app = FaceAnalysis(name="buffalo_l")  # buffalo_l 模型包含属性识别
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 表示用第一块GPU，-1表示CPU
import cv2

img = cv2.imread("/home/rangers/face_workspace/face_insight/facetest.jpeg")
faces = app.get(img)

for face in faces:
    print("性别:", "男" if face.sex == 1 else "女")
    print("年龄:", face.age)
    print("种族:", face.race)   # race 属性（不同版本可能是概率分布，需要自己解析）
    print("置信度:", face.det_score)

