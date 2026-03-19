import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img_grab = cv.imread('coffee_cup.jpg')
if img_grab is None:
    print("과제3 이미지를 찾을 수 없습니다.")
else:
    # 1. 초기화 (배경, 전경 모델 및 마스크 생성)
    mask = np.zeros(img_grab.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 2. 초기 사각형 영역 설정 (x, y, width, height)
    # [경고] 반드시 커피잔 객체가 꽉 차게 들어가는 좌표로 직접 계산해서 변경할 것!
    # 예시: 가로세로 50픽셀씩 여백을 둔 박스
    h, w = img_grab.shape[:2]
    rect = (50, 50, w - 100, h - 100) 

    # 3. GrabCut 실행 (5번 반복)
    cv.grabCut(img_grab, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # 4. 마스크 후처리 (힌트 요구사항 반영)
    # GC_BGD(0), GC_PR_BGD(2)는 확실한 배경이거나 배경일 확률이 높음 -> 0으로 처리
    # GC_FGD(1), GC_PR_FGD(3)은 확실한 전경이거나 전경일 확률이 높음 -> 1로 처리
    mask_binary = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

    # 5. 원본 이미지에서 배경 제거
    # 2D 마스크를 3차원으로 늘려 곱해줍니다.
    extracted_img = img_grab * mask_binary[:, :, np.newaxis]

    # 6. Matplotlib로 세 개 이미지 시각화
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_grab, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # 이진화된 마스크를 흑백으로 출력 (객체는 흰색(1), 배경은 검은색(0))
    plt.imshow(mask_binary, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(extracted_img, cv.COLOR_BGR2RGB))
    plt.title('Extracted Object')
    plt.axis('off')

    plt.show()