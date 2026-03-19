import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_hough = cv.imread('dabo.jpg')
if img_hough is None:
    print("과제2 이미지를 찾을 수 없습니다.")
else:
    gray_hough = cv.cvtColor(img_hough, cv.COLOR_BGR2GRAY)

    # 1. 캐니 에지 검출 (힌트 요구사항: threshold 100, 200 적용)
    edges = cv.Canny(gray_hough, 100, 200)

    # 2. 허프 변환으로 직선 검출
    # 파라미터는 실험적으로 조정 필요 (rho=1, theta=1도, threshold=50, minLineLength=50, maxLineGap=10)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    # 3. 원본 이미지에 빨간색 직선 그리기
    img_hough_drawn = img_hough.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # OpenCV BGR 체계에서 빨간색은 (0, 0, 255), 두께 2
            cv.line(img_hough_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 4. Matplotlib로 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_hough, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_hough_drawn, cv.COLOR_BGR2RGB))
    plt.title('Hough Lines Detected')
    plt.axis('off')

    plt.show()