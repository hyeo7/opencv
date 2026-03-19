import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기 및 그레이스케일 변환
# (경로에 맞게 이미지 파일명을 수정하세요)
img_sobel = cv.imread('edgeDetectionImage.jpg') 
if img_sobel is None:
    print("이미지를 찾을 수 없습니다.")
else:
    gray_sobel = cv.cvtColor(img_sobel, cv.COLOR_BGR2GRAY)

    # 2. x축, y축 방향 소벨 에지 검출 (요구사항: cv.CV_64F, ksize 3 또는 5)
    grad_x = cv.Sobel(gray_sobel, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_sobel, cv.CV_64F, 0, 1, ksize=3)

    # 3. 에지 강도 계산 및 형변환
    magnitude = cv.magnitude(grad_x, grad_y)
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # 4. Matplotlib로 시각화 (원본과 에지 강도 이미지)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    # Matplotlib는 RGB 기반이므로 BGR을 RGB로 변환해야 색상이 깨지지 않습니다.
    plt.imshow(cv.cvtColor(img_sobel, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_uint8, cmap='gray')
    plt.title('Sobel Edge Magnitude')
    plt.axis('off')

    plt.show()