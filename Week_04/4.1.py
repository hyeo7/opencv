import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 영상 읽기 및 그레이스케일 변환
# SIFT 특징점 검출은 명암의 변화량(Gradient)을 기반으로 하므로 컬러 정보는 불필요합니다.
img_sift = cv.imread('mot_color70.jpg') 
if img_sift is None:
    print("이미지를 찾을 수 없습니다. 파일 경로를 확인하세요.")
else:
    gray_sift = cv.cvtColor(img_sift, cv.COLOR_BGR2GRAY)

    # 2. SIFT 객체 생성 및 파라미터 튜닝 [요구사항 반영]
    # SIFT_create()의 매개변수를 조정하여 특징점 검출 결과를 통제합니다.
    # nfeatures: 검출할 최대 특징점의 수를 제한합니다 (특징점이 너무 많아 연산이 느려지는 것 방지).
    # contrastThreshold: 값이 클수록 대비가 낮은 약한 특징점은 무시됩니다 (기본값 0.04).
    # edgeThreshold: 값이 작을수록 에지(Edge)에 위치한 불안정한 특징점을 강하게 걸러냅니다 (기본값 10).
    # sigma: 초기 가우시안 블러링의 강도입니다 (기본값 1.6).
    sift = cv.SIFT_create(nfeatures=4500, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    # 3. 특징점 검출 및 기술자 추출
    # keypoints(kp): 특징점의 좌표, 크기, 방향 정보가 담긴 객체 리스트
    # descriptors(des): 각 특징점 주변의 기울기 분포를 담은 128차원 수치 벡터 행렬
    kp, des = sift.detectAndCompute(gray_sift, None)

    # 4. 특징점 시각화 (크기 및 방향 포함)
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 옵션은 단순한 점이 아니라,
    # 특징점의 크기(Scale, 원의 반지름)와 주 방향(Orientation, 원 내부의 선)을 함께 그려줍니다.
    img_kp = cv.drawKeypoints(img_sift, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 5. Matplotlib을 이용한 시각화
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_sift, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_kp, cv.COLOR_BGR2RGB))
    plt.title(f'SIFT Keypoints')
    plt.axis('off')

    plt.tight_layout()
    plt.show()