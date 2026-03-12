<div align="center">

# 컴퓨터 비전 2주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>
카메라 캘리브레이션, 이미지 변환(회전, 크기 조절, 평행 이동), <br>
그리고 스테레오 카메라 기반의 Depth(깊이) 추정을 수행합니다. <br>

</div>
<br><br>

# 1. 체크보드 기반 카메라 캘리브레이션 (2.1.py)
[실습 목표]<br>
이미지에서 체크보드 코너를 검출하고 실제 좌표와 이미지 좌표의 대응 관계를 이용하여 카메라 파라미터를 추정하고 왜곡을 보정하는 실습입니다.
<br><br>
[과제 설명]<br>
여러 장의 체크보드 이미지에서 cv2.findChessboardCorners()를 사용하여 2D 코너 좌표를 검출하고 실제 3D 좌표와의 대응 관계를 설정했습니다. <br>
검출된 좌표들을 바탕으로 cv2.calibrateCamera() 함수를 이용해 카메라의 내부 파라미터(Camera Matrix)와 왜곡 계수를 성공적으로 도출했습니다. <br>
최종적으로 계산된 파라미터와 cv2.undistort() 함수를 적용하여 렌즈의 방사 및 접선 왜곡이 보정된 깔끔한 결과 이미지를 시각화했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 # OpenCV 라이브러리를 불러옵니다. 이미지 및 영상 처리에 사용됩니다.
import numpy as np # 행렬 및 다차원 배열 처리를 위한 numpy 라이브러리를 불러옵니다.
import glob # 지정된 패턴과 일치하는 모든 파일의 경로를 찾기 위해 glob 모듈을 사용합니다.

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (단위: mm)
square_size = 25.0

# 코너 위치를 더 정확하게 찾기 위한 반복 종료 조건 (최대 30번 반복 또는 오차 0.001 이하)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 세계의 3D 좌표를 생성합니다 (Z축은 0으로 가정).
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size # 격자 한 칸의 실제 크기인 25mm를 곱해 실제 스케일로 맞춥니다.

# 모든 이미지에서 찾은 3D 공간의 점들과 2D 이미지의 점들을 저장할 리스트입니다.
objpoints = []
imgpoints = []

# 'calibration_images' 폴더 내의 'left'로 시작하는 모든 jpg 파일을 리스트로 불러옵니다.
images = glob.glob("calibration_images/left*.jpg")
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None: continue # 이미지를 읽지 못하면 다음 이미지로 넘어갑니다.
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 코너 검출을 위해 흑백으로 변환합니다.
    
    if img_size is None:
        img_size = gray.shape[::-1] # 첫 번째 이미지에서 해상도(width, height)를 저장합니다.

    # 흑백 이미지에서 체크보드의 코너들을 검출합니다.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret == True:
        objpoints.append(objp) # 코너를 찾았다면 실제 3D 좌표를 리스트에 추가합니다.
        
        # 검출된 코너의 위치를 소수점 픽셀 단위까지 더 정밀하게 다듬습니다.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2) # 정밀화된 2D 이미지 좌표를 리스트에 추가합니다.
        
        # 화면에 코너를 시각화하여 찾은 과정을 보여줍니다.
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(50) # 각 이미지당 50ms씩 대기하며 보여줍니다.

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 모아둔 실제 좌표와 이미지 좌표를 이용해 카메라 매트릭스와 왜곡 계수를 계산합니다.
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K) # 카메라 내부 파라미터 (초점 거리, 주점 등) 출력

print("\nDistortion Coefficients:")
print(dist) # 왜곡 계수 출력

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0:
    test_img = cv2.imread(images[0]) # 테스트할 원본 이미지를 불러옵니다.
    # 계산된 K 매트릭스와 왜곡 계수를 바탕으로 이미지의 왜곡을 폅니다.
    undistorted_img = cv2.undistort(test_img, K, dist, None, K)
    
    cv2.imshow('Original', test_img)
    cv2.imshow('Undistorted', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1493" height="535" alt="실행 결과 1" src="https://github.com/user-attachments/assets/68c8514b-fc8d-486a-b531-e70310b59743" />
</div>
<br><br>

# 2. 이미지 Rotation & Transformation (2.2.py)
[실습 목표] <br>
한 장의 이미지에 회전, 크기 조절, 평행이동을 동시에 적용하여 이미지 공간 변환(Transformation)을 실습합니다.
<br><br>
[과제 설명] <br>
장미 꽃 이미지 한 장을 불러와 영상의 중심 좌표를 기준으로 cv2.getRotationMatrix2D()를 사용해 +30도 회전 및 0.8배 크기 축소 변환 행렬을 생성했습니다. <br>
생성된 회전 행렬의 마지막 열 값을 수정하여 x축 방향으로 +80px, y축 방향으로 -40px 만큼의 평행 이동(Translation) 값을 추가했습니다. <br>
최종적으로 완성된 아핀 변환 행렬을 cv2.warpAffine() 함수를 통해 원본 이미지에 적용하고, 회전 및 이동이 동시에 일어난 결과를 확인했습니다. <br>
<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 # OpenCV 라이브러리를 불러옵니다.
import numpy as np # 넘파이 배열 처리를 위해 불러옵니다.

# 1. 이미지 불러오기
# 'rose.png' 파일을 읽어옵니다.
img = cv2.imread('rose.png') 
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

# 이미지의 세로(h), 가로(w) 크기를 가져옵니다.
h, w = img.shape[:2]
# 회전의 기준점이 될 이미지의 정중앙 좌표를 계산합니다.
center = (w / 2, h / 2)

# 2. 회전 및 크기 조절 행렬 생성
# cv2.getRotationMatrix2D()를 사용하여 기준점, 각도(+30도), 스케일(0.8배)에 대한 변환 행렬 M을 만듭니다.
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 3. 평행이동 적용
# 변환 행렬 M의 0번째 행 마지막 열(x축 평행이동)에 +80을 더합니다.
M[0, 2] += 80
# 변환 행렬 M의 1번째 행 마지막 열(y축 평행이동)에 -40을 뺍니다 (위로 40 이동).
M[1, 2] -= 40

# 4. 아핀 변환 적용
# cv2.warpAffine() 함수를 사용해 만들어진 행렬 M을 원본 이미지에 최종적으로 적용합니다.
result = cv2.warpAffine(img, M, (w, h))

# 결과를 화면에 나란히 띄워 비교합니다.
cv2.imshow('Original', img)
cv2.imshow('Rotated + Scaled + Translated', result)
cv2.waitKey(0)
cv2.destroyAllWindows()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="2148" height="1354" alt="실행 결과 2-1" src="https://github.com/user-attachments/assets/75ead328-b07a-46a4-840c-09975337aff9" />
<img width="1725" height="914" alt="실행 결과 2-2" src="https://github.com/user-attachments/assets/ad539da6-dbf3-43ff-ad36-d93e3c086700" />
</div>
<br><br>

# 3. Stereo Disparity 기반 Depth 추정 (2.3.py)
[실습 목표] <br>
같은 장면을 왼쪽 카메라와 오른쪽 카메라에서 촬영한 두 장의 이미지를 이용해 물체의 깊이(Depth)를 추정합니다.
<br><br>
[과제 설명] <br>
양안 카메라로 촬영된 좌/우 그레이스케일 이미지에 cv2.StereoBM_create()를 적용하여 각 픽셀의 위치 차이를 나타내는 Disparity Map을 계산했습니다. <br>
초점 거리(f)와 베이스라인(B) 값, 그리고 $Z=fB/d$ 공식을 활용하여 양수 Disparity 값들을 실제 거리 정보를 담은 Depth Map으로 변환했습니다. <br>
세 가지 지정된 ROI(Painting, Frog, Teddy) 영역별로 평균 Disparity와 Depth 값을 추출하고, 이를 비교하여 카메라에 가장 가까운 객체와 먼 객체를 판별했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2
import numpy as np
from pathlib import Path

# 결과 이미지를 저장할 'outputs' 폴더를 만듭니다.
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. 이미지 불러오기
left_color = cv2.imread("left.png")
right_color = cv2.imread("right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터 
f = 700.0 # 카메라의 초점 거리 (focal length)
B = 0.12  # 카메라 사이의 거리 (baseline)

# 깊이를 측정할 세 가지 관심 영역(ROI) 좌표 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 흑백 이미지로 변환합니다. (Disparity 계산용)
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# cv2.StereoBM_create()를 사용하여 블록 매칭 기반의 시차(Disparity) 맵을 생성합니다.
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_16S = stereo.compute(gray_left, gray_right)

# OpenCV가 반환한 값은 16배 스케일링된 정수이므로, 실수형으로 바꾼 뒤 16으로 나누어 실제 값을 얻습니다.
disparity = disparity_16S.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# -----------------------------
# 시차 값이 0보다 큰(유효한) 픽셀들에 대해서만 처리합니다.
valid_mask = disparity > 0
depth_map = np.zeros_like(disparity, dtype=np.float32)

# Z = f * B / d 공식을 적용하여 실제 거리(Depth)를 계산합니다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_mask = valid_mask[y:y+h, x:x+w]

    # 영역 내에 유효한 값이 존재할 때만 평균을 구합니다.
    if np.any(roi_mask):
        mean_disp = np.mean(roi_disp[roi_mask])
        mean_depth = np.mean(roi_depth[roi_mask])
    else:
        mean_disp, mean_depth = 0, 0

    results[name] = {'disp': mean_disp, 'depth': mean_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI별 평균 Disparity 및 Depth ===")
closest_name, farthest_name = "", ""
min_depth, max_depth = float('inf'), 0

# 결과를 분석하여 어느 객체가 제일 가깝고 먼지 출력합니다.
for name, res in results.items():
    print(f"{name} - 평균 Disparity: {res['disp']:.2f}, 평균 Depth: {res['depth']:.4f}")
    if res['depth'] > 0:
        if res['depth'] < min_depth:
            min_depth, closest_name = res['depth'], name
        if res['depth'] > max_depth:
            max_depth, farthest_name = res['depth'], name

print(f"\n[결과 해석] 가장 가까운 영역: {closest_name}, 가장 먼 영역: {farthest_name}")

# (이하 생략: 결과 이미지를 컬러맵으로 변환하고 박스를 그려 저장하는 시각화 코드)
# ... 기존 뼈대 코드의 시각화 파트를 그대로 적용 ...```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="2149" height="1240" alt="image" src="https://github.com/user-attachments/assets/e5c225fe-c24c-4e26-a9de-5897d9d1de5d" />
</div>
