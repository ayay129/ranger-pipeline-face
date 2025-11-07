#!/usr/bin/env python3
"""
Test script to verify gender, age, and race recognition in InsightFace implementation
"""

import os
import cv2
import numpy as np
from face_model_insightface import FaceModel

# 设置日志
def log(message):
    print(f"[TEST] {message}")

# 加载测试图像
def load_test_image(image_path='./facetest.jpeg'):
    if not os.path.exists(image_path):
        # 如果默认图像不存在，创建一个简单的测试图像
        log(f"Test image {image_path} not found, creating a test image")
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色背景
        # 绘制一个简单的人脸轮廓
        cv2.ellipse(img, (320, 240), (100, 130), 0, 0, 360, (0, 0, 0), 2)
        # 绘制眼睛、鼻子和嘴巴
        cv2.circle(img, (270, 210), 15, (0, 0, 0), -1)
        cv2.circle(img, (370, 210), 15, (0, 0, 0), -1)
        cv2.circle(img, (320, 270), 10, (0, 0, 0), -1)
        cv2.ellipse(img, (320, 320), (40, 20), 0, 0, 180, (0, 0, 0), 2)
        return img
    
    log(f"Loading test image from {image_path}")
    return cv2.imread(image_path)

# 测试性别、年龄和种族识别功能
def test_attributes_recognition():
    log("Initializing FaceModel...")
    model = FaceModel()
    
    log("Loading test image...")
    img = load_test_image()
    
    log("Performing face detection and feature extraction...")
    num_faces, bboxes, landmarks, aligned_faces = model.get_input(img)
    
    log(f"Detected {num_faces} faces")
    
    if num_faces > 0:
        log("Extracting face features and predicting attributes...")
        
        # 存储所有识别结果
        results = []
        
        for i in range(num_faces):
            # 获取人脸特征
            feature = model.get_feature(aligned_faces[i])
            
            # 获取性别、年龄和种族
            gender, age, race = model.get_gender_age_race(aligned_faces[i])
            
            # 存储结果
            results.append({
                'face_index': i,
                'bbox': bboxes[i].tolist(),
                'gender': gender,
                'age': age,
                'race': race
            })
        
        # 打印结果
        log("Attribute recognition results:")
        for result in results:
            log(f"Face {result['face_index']}:")
            log(f"  BBox: {result['bbox']}")
            log(f"  Gender: {result['gender']}")
            log(f"  Age: {result['age']}")
            log(f"  Race: {result['race']}")
        
        # 验证结果是否合理
        all_unknown = all(r['gender'] == 'Unknown' and r['race'] == 'Unknown' and r['age'] == 0 for r in results)
        if all_unknown:
            log("ERROR: All attributes are Unknown. The fix might not be working correctly.")
            return False
        
        log("SUCCESS: Attributes are being recognized correctly!")
        return True
    else:
        log("No faces detected. Cannot test attribute recognition.")
        return False

# 运行测试
def run_tests():
    log("Starting InsightFace attribute recognition tests...")
    success = test_attributes_recognition()
    
    if success:
        log("\nAll tests passed successfully!")
        log("You can now start the server with the following command:")
        log("python face_fastapi_insightface.py")
        log("Then access the API and set gender=1, age=1, race=1 to verify the fix.")
    else:
        log("\nSome tests failed. Please check the implementation.")

if __name__ == '__main__':
    run_tests()
