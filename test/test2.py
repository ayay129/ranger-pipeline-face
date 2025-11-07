import insightface
model = insightface.model_zoo.get_model('genderage')
model.prepare(ctx_id=0)

import cv2

img = cv2.imread("/home/rangers/face_workspace/face_insight/facetest.jpeg")
rimg = model.get(img)  # img 是输入图像
for face in rimg:
    print(face['gender'], face['age'])

