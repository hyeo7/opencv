import cv2
import numpy as np

# 장미 이미지 불러오기 (경로는 폴더 구조에 맞게 수정해주세요)
img = cv2.imread('rose.png') 
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

h, w = img.shape[:2]
center = (w / 2, h / 2)

# cv2.getRotationMatrix2D()를 사용하여 회전 및 크기 조절 행렬 생성 [cite: 1330]
# 30도 회전, 크기 0.8 조절 
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동 
M[0, 2] += 80
M[1, 2] -= 40

# cv2.warpAffine()로 변환 적용 [cite: 1331]
result = cv2.warpAffine(img, M, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Rotated + Scaled + Translated', result)
cv2.waitKey(0)
cv2.destroyAllWindows()