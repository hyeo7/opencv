<div align="center">

# 컴퓨터 비전 4주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>
SIFT 기반 스케일 불변 특징점 검출, FLANN 및 BFMatcher를 이용한 특징점 매칭, <br>
그리고 RANSAC 기반 호모그래피(Homography) 추정을 통한 파노라마 정합 실습입니다. <br>

</div>
<br><br>

# 1. SIFT를 이용한 특징점 검출 및 시각화 (4.1.py)
[실습 목표]<br>
주어진 이미지에서 회전과 스케일 변화에 강인한 SIFT 특징점을 검출하고, 검출된 좌표의 크기와 방향성을 시각화하는 실습입니다.
<br><br>
[과제 설명]<br>
cv.SIFT_create() 객체를 생성할 때, nfeatures, contrastThreshold, edgeThreshold 등의 파라미터를 제어하여 <br> 
노이즈 픽셀이나 에지(Edge)에 위치한 불안정한 점들을 수학적으로 걸러냈습니다. <br>
detectAndCompute() 연산을 통해 영상 피라미드 기반의 스케일 공간에서 극대/극소점을 찾아내어 <br>
크기, 방향 정보가 담긴 특징점(Keypoint)과 128차원의 기술자(Descriptor)를 성공적으로 확보했습니다. <br>
cv.drawKeypoints() 함수에 DRAW_RICH_KEYPOINTS 플래그를 적용해 단순한 점이 아닌, 각 특징점의 기하학적 스케일(반경)과 <br>
주 방향성(Orientation)을 원본 이미지 위에 명확히 시각화하여 Matplotlib로 나란히 비교 출력했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
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
    plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1786" height="631" alt="image" src="https://github.com/user-attachments/assets/9a340297-1daf-427f-82ac-a2626928b4b8" />
</div>
<br><br>

# 2. SIFT를 이용한 두 영상 간 특징점 매칭 (4.2.py)
[실습 목표] <br>
두 개의 연속된 프레임 이미지에서 각각 SIFT 특징점을 추출하고, <br>
두 점군 사이의 128차원 기술자(Descriptor) 거리를 비교하여 동일한 객체(대응점)를 추적하는 실습입니다.
<br><br>
[과제 설명] <br>
서로 다른 두 프레임 이미지 전체와 크롭된 특정 영역(버스) 간의 추적을 위해 SIFT 특징점을 각각 추출했습니다. <br>
128차원 공간에서 유클리디안 거리를 빠르게 탐색하는 cv.DescriptorMatcher_FLANNBASED 매처를 사용하여 최근접 이웃(knnMatch) 2개를 찾아냈습니다. <br>
단일 최소 거리 매칭이 유발하는 극심한 오매칭(False Positive)을 막기 위해, <br>
1순위와 2순위 거리의 비율을 비교하는 Lowe의 비율 테스트를 적용하여 헷갈리는 후보가 없다고 판단되는 정답 쌍만 선별했습니다. <br>
선별된 매칭점만을 cv.drawMatches()를 통해 연결선으로 시각화하였습니다. <br>
<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
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
cv.destroyAllWindows()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1606" height="815" alt="image" src="https://github.com/user-attachments/assets/97adb3cd-8312-4d83-ac41-29af1390ffc6" />
<img width="499" height="57" alt="image" src="https://github.com/user-attachments/assets/b1906899-c19f-4c74-9a6e-25eb25e439b2" />
</div>
<br><br>

# 3. 호모그래피를 이용한 파노라마 이미지 정합 (4.3.py)
[실습 목표] <br>
SIFT 특징점 매칭을 통해 두 이미지 간의 기하학적 대응점을 찾고, <br>
RANSAC 기반 호모그래피를 추정하여 파노라마 캔버스 형태로 이미지를 이어 붙이는 투시 변환 실습입니다.
<br><br>
[과제 설명] <br>
전수 조사 방식인 cv.BFMatcher(cv.NORM_L2)와 KNN, Ratio Test(0.7)를 결합하여 SIFT 기술자 간의 대응점을 1차적으로 추출했습니다.. <br>
1차 필터링된 대응점 좌표들에 cv.findHomography()와 RANSAC 알고리즘을 결합하여, 남은 Outlier의 영향을 완벽히 배제한 최적의 3x3 투시 변환 행렬($H$)을 추정했습니다. <br>
cv.warpPerspective() 연산 시 기준 좌표계가 음수(-) 영역으로 넘어가 왼쪽 픽셀이 소실되는 클리핑(Clipping) 현상을 막기 위해, <br>
오른쪽 이미지를 왼쪽 기준계로 당겨오는 기하학적 파이프라인 역전을 설계했습니다. <br>
최종적으로 (w1+w2, max(h1,h2)) 크기의 캔버스에서 RANSAC 검증을 통과한 매칭선(Inlier) 시각화와, 정교하게 결합된 파노라마 정합 결과를 도출했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
# 요구사항: cv.imread()를 사용하여 두 개의 이미지를 불러옴
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

if img1 is None or img2 is None:
    print("이미지를 찾을 수 없습니다. 파일 경로 및 이름을 확인하십시오.")
else:
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 2. SIFT 특징점 검출
    # 요구사항: Cv.SIFT_create()를 사용하여 특징점을 검출
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. BFMatcher 특징점 매칭
    # 요구사항: cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭
    bf = cv.BFMatcher(cv.NORM_L2)
    knn_match = bf.knnMatch(des1, des2, k=2)

    # 4. 좋은 매칭점 선별 (Ratio Test)
    # 요구사항 힌트: 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별
    good_match = []
    for m, n in knn_match:
        if m.distance < 0.7 * n.distance:
            good_match.append(m)

    # 5. 호모그래피 좌표 배열 추출
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

    # 6. [핵심 교정] 투시 변환 방향 역전 (points2 -> points1)
    # 오른쪽 이미지(img2)를 왼쪽 이미지(img1) 좌표계에 맞추기 위해 파라미터 순서를 변경합니다.
    # 요구사항 힌트: cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음
    H, mask = cv.findHomography(points2, points1, cv.RANSAC, 5.0)

    # 7. 이미지 투시 변환 및 정합
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 요구사항 힌트: 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
    panorama_w = w1 + w2
    panorama_h = max(h1, h2)

    # 요구사항: cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
    # img2를 H 행렬에 따라 투시 변환합니다 (양수 X축 방향으로 이동됨).
    warped_img2 = cv.warpPerspective(img2, H, (panorama_w, panorama_h))
    
    # 변환된 캔버스의 (0,0) 원점 위치에 왼쪽 이미지인 img1을 덮어씌웁니다.
    panorama_result = warped_img2.copy()
    panorama_result[0:h1, 0:w1] = img1

    # 매칭 결과 시각화 준비
    matches_mask = mask.ravel().tolist()
    # 주의: findHomography에서 points2 -> points1 순서로 계산했으므로, 
    # drawMatches에서도 img2와 img1의 순서를 맞춰 그려주어야 매칭선이 꼬이지 않습니다.
    img_match = cv.drawMatches(img2, kp2, img1, kp1, good_match, None, 
                               matchesMask=matches_mask, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 8. Matplotlib 시각화
    # 요구사항: 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    plt.title('Matching Result (RANSAC Inliers)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(panorama_result, cv.COLOR_BGR2RGB))
    plt.title('Warped Image (Panorama Alignment)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="2352" height="453" alt="image" src="https://github.com/user-attachments/assets/1d471add-ee21-4c5e-98d8-628b2d51fba8" />
</div>

