import cv2
import numpy as np

# 1. 이미지 불러오기
img_color = cv2.imread('soccer.jpg')
img_gray = cv2.imread('soccer.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 이미지 크기
width = 500
height = int(img_color.shape[0] * (width / img_color.shape[1]))
img_color_resized = cv2.resize(img_color, (width, height))
img_gray_resized = cv2.resize(img_gray, (width, height))

# 3. 흑백 이미지를 3채널로 변환 
img_gray_3ch = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2BGR)

# 4. 이미지 가로로 나란히 붙이기
result = np.hstack((img_color_resized, img_gray_3ch))

# 5. 결과 창에 출력
cv2.imshow('Color and Grayscale', result)
cv2.waitKey(0)
cv2.destroyAllWindows()