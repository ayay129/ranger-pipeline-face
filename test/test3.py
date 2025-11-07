import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")  # buffalo_l 包含性别/年龄/种族
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread("/home/rangers/face_workspace/face_insight/facetest.jpeg")
faces = app.get(img)

for f in faces:
    print("性别:", "男" if f.sex == 1 else "女")
    print("年龄:", f.age)
    print("种族概率:", f.race_probs)  # [Asian, White, Black, Indian]

