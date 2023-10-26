from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

def popartify(image_path, num_colors):
    # 이미지 불러오기
    image = cv2.imread(image_path)

    # 이미지 크기 조정 (원하는 크기로)
    image = cv2.resize(image, (600, 600))

    # K-Means 클러스터링을 사용하여 이미지 색상을 줄임
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()].reshape((image.shape))

    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        if image:
            image.save("static/images/uploaded_image.jpg")

            # 팝아트로 변환
            num_colors = 4  # 팝아트 스타일에 사용할 색상 수
            popart_image = popartify("static/images/uploaded_image.jpg", num_colors)
            cv2.imwrite("static/images/popart_image.jpg", popart_image)

            return send_from_directory("static/images", "popart_image.jpg")
    return "이미지를 업로드하세요."

if __name__ == '__main__':
    app.run(debug=True)
