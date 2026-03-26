import cv2 as cv
import numpy as np
import time

# 1. 모델 영상(Query)과 장면 영상(Train) 읽기
# [핵심] img1 전체를 쓰지 않고 배열 슬라이싱을 통해 70번째 프레임에서 '버스'만 오려냅니다.
img1 = cv.imread('mot_color70.jpg')[190:350, 440:560]
if img1 is None:
    print("mot_color70.jpg 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
    exit()
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv.imread('mot_color83.jpg')
if img2 is None:
    print("mot_color83.jpg 이미지를 찾을 수 없습니다.")
    exit()
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 특징점 검출 및 기술자 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# 특징점 개수를 터미널에 출력합니다.
print('특징점 개수: ', len(kp1), len(kp2))

# 3. FLANN 기반 매칭 및 소요 시간 측정
start = time.time() # 매칭 알고리즘 시작 시간 기록

# cv.DescriptorMatcher_FLANNBASED 옵션을 통해 고속 근사 최근접 이웃 탐색기를 생성합니다.
flann_matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
# 각 특징점 당 가장 가까운 이웃 2개를 찾습니다.
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 4. 최근접 이웃 거리 비율(Ratio Test) 적용
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    # 1순위 거리가 2순위 거리보다 월등히(0.7배 미만) 가까울 때만 유효한 매칭으로 인정합니다.
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# 매칭 연산에 걸린 총 소요 시간을 출력합니다.
print('매칭에 걸린 시간: ', time.time() - start)

# 5. 매칭 결과 시각화
# 첨부하신 사진처럼 왼쪽 상단에 버스 이미지를, 우측에 전체 화면을 배치하기 위한 빈 캔버스를 만듭니다.
img_match = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

# 걸러진 good_match 쌍들을 선으로 이어 캔버스에 그립니다.
cv.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Matplotlib 대신 cv.imshow를 사용하여 별도의 윈도우 창으로 결과를 띄웁니다.
cv.imshow('Good Matches', img_match)
cv.waitKey()
cv.destroyAllWindows()