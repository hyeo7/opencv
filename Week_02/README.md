<div align="center">

# 컴퓨터 비전 2주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>
카메라 캘리브레이션, 이미지 변환(회전, 크기 조절, 평행 이동), <br>
그리고 스테레오 카메라 기반의 Depth(깊이) 추정하는 실습입니다. <br>

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
import cv2 # 영상 처리를 위한 OpenCV 라이브러리를 가져옵니다.
import numpy as np # 행렬 및 수학 연산을 위한 NumPy 라이브러리를 가져옵니다.
import glob # 폴더 내의 파일 경로들을 패턴으로 찾기 위해 glob 라이브러리를 가져옵니다.

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6) # 검출할 체스보드의 가로, 세로 내부 코너 개수를 튜플로 정의합니다.

# 체크보드 한 칸 실제 크기 (단위: mm)
square_size = 25.0 # 체스보드 격자 한 칸의 실제 물리적 크기(25mm)를 설정합니다.

# 코너 위치를 서브픽셀(소수점) 단위로 정밀하게 찾기 위한 알고리즘 종료 조건을 설정합니다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

# 실제 세계의 3D 좌표를 저장할 빈 배열을 생성합니다 (Z축은 0으로 가정하므로 평면 좌표계입니다).
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32) 

# np.mgrid를 사용하여 체스보드의 x, y 격자 인덱스를 생성하고 2열 행렬로 형태를 바꿉니다.
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 

# 생성된 인덱스에 격자 한 칸의 실제 크기를 곱해 실제 스케일(mm)이 반영된 3D 좌표를 완성합니다.
objp *= square_size 

# 캘리브레이션에 사용할 모든 이미지의 실제 3D 좌표들을 모아둘 빈 리스트를 만듭니다.
objpoints = [] 

# 캘리브레이션에 사용할 모든 이미지의 2D 픽셀 좌표들을 모아둘 빈 리스트를 만듭니다.
imgpoints = [] 

# 'calibration_images' 폴더 안에 있는 'left'로 시작하는 모든 jpg 파일의 경로를 리스트로 불러옵니다.
images = glob.glob("calibration_images/left*.jpg") 

# 이미지의 해상도(가로, 세로 크기)를 저장할 변수를 초기화합니다.
img_size = None 

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images: # glob으로 찾은 파일 경로 리스트를 하나씩 반복하며 처리합니다.
    img = cv2.imread(fname) # 해당 경로의 이미지 파일을 컬러로 읽어옵니다.
    if img is None: # 만약 이미지를 정상적으로 읽지 못했다면
        continue # 이번 반복을 건너뛰고 다음 파일로 넘어갑니다.
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 코너 검출을 수월하게 하기 위해 이미지를 흑백으로 변환합니다.
    
    if img_size is None: # 만약 이미지 크기가 아직 저장되지 않았다면 (첫 번째 반복일 때)
        img_size = gray.shape[::-1] # 흑백 이미지의 차원(행, 열)을 뒤집어 (가로, 세로) 형태로 저장합니다.

    # 흑백 이미지에서 설정한 체스보드 크기(9x6)에 맞는 코너점들을 찾습니다.
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None) 
    
    if ret == True: # 만약 코너를 성공적으로 모두 찾았다면
        objpoints.append(objp) # 준비해둔 실제 3D 좌표 세트를 objpoints 리스트에 추가합니다.
        
        # 검출된 정수 단위의 코너점(corners) 위치를 소수점 단위까지 더 정밀하게 다듬어 corners2에 저장합니다.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) 
        
        imgpoints.append(corners2) # 정밀화된 2D 이미지 좌표를 imgpoints 리스트에 추가합니다.
        
        # 원본 컬러 이미지 위에 찾은 코너점들을 알록달록한 선과 점으로 그려 넣습니다.
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret) 
        
        cv2.imshow('img', img) # 코너가 그려진 이미지를 'img'라는 창에 띄워 보여줍니다.
        cv2.waitKey(50) # 화면에 띄운 뒤 50 밀리초 동안 잠시 대기합니다.

cv2.destroyAllWindows() # 모든 이미지의 처리가 끝나면 띄워두었던 코너 확인용 창을 모두 닫습니다.

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 모아둔 실제 3D 좌표(objpoints)와 2D 픽셀 좌표(imgpoints)를 바탕으로 카메라 파라미터를 계산합니다.
ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None, # cameraMatrix를 처음 계산하므로 비워둡니다.
    None  # distCoeffs를 처음 계산하므로 비워둡니다.
)

# 계산된 카메라 내부 파라미터(Camera Matrix) 행렬을 출력합니다.
# 3x3 행렬 구조: 
# [ fx,  0, px ]
# [  0, fy, py ]
# [  0,  0,  1 ]
print("카메라 내부 파라미터 행렬 [3x3] (Camera Matrix k):") 
print(k) 
# f (fx, fy): 카메라의 focal length. 카메라가 이미지를 픽셀 단위로 얼마나 확대해서 찍는지를 나타냅니다.
# p (px, py): principal point. 렌즈의 중심이 이미지 센서에 투영되는(찍히는) 중심 좌표를 의미합니다.

# 계산된 렌즈 왜곡 계수(Distortion Coefficients) 5개 요소를 출력합니다.
print("\n왜곡 계수 [k1, k2, p1, p2, k3] (Distortion Coefficients):") 
print(dist) 
# k1, k2, k3: 방사 왜곡(Radial Distortion). 렌즈 중심에서 멀어질수록 직선이 바깥이나 안으로 휘어지는 현상입니다.
# p1, p2: 접선 왜곡(Tangential Distortion). 렌즈와 이미지 센서가 물리적으로 완전히 평행하지 않아서 발생하는 현상입니다.

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
if len(images) > 0: # 처리한 이미지가 1장이라도 있다면 왜곡 보정 테스트를 진행합니다.
    test_img = cv2.imread(images[0]) # 리스트의 첫 번째 이미지를 테스트용 원본 이미지로 다시 읽어옵니다.
    
    # 원본 이미지에 계산된 내부 행렬(k)과 왜곡 계수(dist)를 적용하여 방사/접선 왜곡이 펴진 새 이미지를 만듭니다.
    undistorted_img = cv2.undistort(test_img, k, dist, None, k) 
    
    cv2.imshow('Original', test_img) # 왜곡 보정 전의 원본 이미지를 'Original' 창에 띄웁니다.
    cv2.imshow('Undistorted', undistorted_img) # 왜곡이 수학적으로 보정된 이미지를 'Undistorted' 창에 띄웁니다.
    cv2.waitKey(0) # 사용자가 아무 키나 누를 때까지 창을 닫지 않고 무한정 대기합니다.
    cv2.destroyAllWindows() # 키 입력이 들어오면 모든 창을 닫고 프로그램을 종료합니다.```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1909" height="760" alt="image" src="https://github.com/user-attachments/assets/55fc9b8b-4154-4685-bca8-f448c1a57d3c" />
<img width="724" height="182" alt="image" src="https://github.com/user-attachments/assets/b31a2dac-d9b5-48b1-9803-f4aced5057c7" />
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
import cv2 # 영상 처리를 위한 OpenCV 라이브러리를 가져옵니다.
import numpy as np # 행렬 연산을 처리하기 위해 NumPy 라이브러리를 가져옵니다.

# 1. 이미지 불러오기
img = cv2.imread('rose.png') # 현재 폴더에 있는 'rose.png' 파일을 컬러 이미지로 읽어와 img 변수에 저장합니다.

if img is None: # 만약 해당 경로에 파일이 없어 이미지를 불러오지 못했다면
    print("이미지를 불러올 수 없습니다.") # 에러 메시지를 출력합니다.
    exit() # 프로그램 실행을 즉시 강제로 종료합니다.

# img.shape는 (세로, 가로, 채널) 정보를 담고 있으므로 앞의 두 값(세로, 가로)만 가져와 h, w 변수에 저장합니다.
h, w = img.shape[:2] 

# 회전의 기준점이 될 이미지의 정중앙 좌표를 구하기 위해 가로와 세로를 2로 나누어 튜플로 저장합니다.
center = (w / 2, h / 2) 

# 2. 회전 및 크기 조절 행렬 생성
# cv2.getRotationMatrix2D 함수를 사용하여 변환 행렬 M을 만듭니다.
# 첫 번째 인자(center): 회전 기준점, 두 번째 인자(30): 반시계 방향으로 30도 회전, 세 번째 인자(0.8): 0.8배 축소
M = cv2.getRotationMatrix2D(center, 30, 0.8) 

# 3. 평행이동 적용
# 2x3 변환 행렬 M의 0번째 행, 2번째 열의 값(X축 평행이동)에 +80을 더합니다. (오른쪽으로 80픽셀 이동)
M[0, 2] += 80 

# 변환 행렬 M의 1번째 행, 2번째 열의 값(Y축 평행이동)에 -40을 뺍니다. (위쪽으로 40픽셀 이동)
M[1, 2] -= 40 

# 4. 아핀 변환 적용
# cv2.warpAffine 함수에 원본 이미지(img)와 완성된 변환 행렬(M), 그리고 결과 이미지의 크기(w, h)를 넣어 변환을 실행합니다.
result = cv2.warpAffine(img, M, (w, h)) 

cv2.imshow('Original', img) # 변환 전 원본 이미지를 'Original'이라는 창에 띄워줍니다.
cv2.imshow('Rotated + Scaled + Translated', result) # 회전, 크기조절, 평행이동이 모두 적용된 결과 이미지를 띄워줍니다.
cv2.waitKey(0) # 사용자가 키보드를 누를 때까지 창을 끄지 않고 유지합니다.
cv2.destroyAllWindows() # 키보드 입력이 감지되면 생성된 모든 OpenCV 창을 닫고 메모리를 정리합니다.```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1178" height="427" alt="image" src="https://github.com/user-attachments/assets/5604392a-f5b1-464f-9f05-c4499ce0cd94" />
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
import cv2 # 영상 처리를 위한 OpenCV 라이브러리를 가져옵니다.
import numpy as np # 수치 계산과 배열 처리를 위한 NumPy 라이브러리를 가져옵니다.
from pathlib import Path # 파일 경로 및 폴더 생성을 쉽게 다루기 위해 pathlib 라이브러리의 Path를 가져옵니다.

# 결과를 저장할 'outputs'라는 이름의 폴더 경로 객체를 만듭니다.
output_dir = Path("./outputs") 

# 'outputs' 폴더가 없다면 새로 만들고, 이미 존재한다면 에러 없이 무시하도록(exist_ok=True) 설정합니다.
output_dir.mkdir(parents=True, exist_ok=True) 

# 1. 좌/우 이미지 불러오기
left_color = cv2.imread("left.png") # 왼쪽 카메라에서 찍은 컬러 이미지를 불러옵니다.
right_color = cv2.imread("right.png") # 오른쪽 카메라에서 찍은 컬러 이미지를 불러옵니다.

if left_color is None or right_color is None: # 두 이미지 중 하나라도 불러오지 못했다면
    print("❌ 이미지를 찾지 못했습니다! 파일 경로를 확인해 주세요.") # 경고 메시지를 출력합니다.
    exit() # 프로그램 실행을 종료합니다.

# 카메라 고유 파라미터 설정
f = 700.0 # 카메라 렌즈의 초점 거리(focal length)를 700.0으로 설정합니다.
B = 0.12 # 두 카메라 렌즈 사이의 물리적 거리(Baseline)를 0.12m로 설정합니다.

# 깊이를 분석할 3개의 관심 영역(ROI)을 딕셔너리로 정의합니다. 값은 (x, y, width, height) 순서입니다.
rois = {
    "Painting": (55, 50, 130, 110), # 왼쪽 위 그림 액자 영역 (가장 멀리 있음)
    "Frog": (90, 265, 230, 95), # 아래쪽 초록색 개구리 인형 영역 (중간 거리)
    "Teddy": (310, 35, 115, 90) # 오른쪽 위 분홍색 곰 인형 영역 (가장 가까이 있음)
}

# 시차(Disparity)를 계산하는 알고리즘은 흑백 이미지를 사용하므로, 왼쪽 이미지를 흑백으로 변환합니다.
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY) 

# 오른쪽 이미지도 동일하게 흑백 이미지로 변환해 줍니다.
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY) 

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# 좌우 이미지 간 픽셀을 비교하는 블록 매칭 기반의 StereoBM 객체를 만듭니다.
# numDisparities=64(탐색할 최대 픽셀 이동 범위), blockSize=15(비교할 픽셀 블록의 크기)로 설정합니다.
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15) 

# 만들어진 stereo 객체에 좌우 흑백 이미지를 넣어 각 픽셀별 시차 값(16배 곱해진 정수)을 계산합니다.
disparity_16S = stereo.compute(gray_left, gray_right) 

# 계산된 정수값을 실수형(float32)으로 형변환한 뒤 16으로 나누어 실제 픽셀 단위의 시차(Disparity) 값을 얻습니다.
disparity = disparity_16S.astype(np.float32) / 16.0 

# -----------------------------
# 2. Depth 계산 (Z = fB / d)
# -----------------------------
# 시차 값이 0보다 큰(양수) 부분만 실제 매칭이 성공한 유효한 픽셀로 간주하기 위해 마스크(조건)를 만듭니다.
valid_mask = disparity > 0 

# 시차 맵과 크기가 같은 0으로 채워진 빈 실수형 행렬(Depth 맵)을 생성합니다.
depth_map = np.zeros_like(disparity, dtype=np.float32) 

# 매칭이 성공한 유효한 픽셀 위치에 한해, Z = (f * B) / d 공식을 적용하여 실제 거리(Depth)를 계산하여 넣습니다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask] 

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {} # 각 영역별 평균 결과값을 저장할 빈 딕셔너리를 만듭니다.

for name, (x, y, w, h) in rois.items(): # 설정해둔 3개의 관심 영역(ROI)을 순서대로 하나씩 꺼내어 반복합니다.
    roi_disp = disparity[y:y+h, x:x+w] # 전체 시차 맵에서 현재 ROI 영역에 해당하는 부분만 잘라냅니다.
    roi_depth = depth_map[y:y+h, x:x+w] # 전체 깊이 맵에서 현재 ROI 영역에 해당하는 부분만 잘라냅니다.
    roi_mask = valid_mask[y:y+h, x:x+w] # 유효 픽셀 마스크에서도 현재 ROI 영역 부분만 잘라냅니다.

    if np.any(roi_mask): # 잘라낸 영역 안에 매칭에 성공한 유효한 픽셀이 하나라도 존재한다면
        mean_disp = np.mean(roi_disp[roi_mask]) # 유효한 시차 값들만 모아서 평균을 구합니다.
        mean_depth = np.mean(roi_depth[roi_mask]) # 유효한 깊이 값들만 모아서 평균을 구합니다.
    else: # 영역 내에 유효한 픽셀이 전혀 없다면 (검은색 0으로만 채워졌다면)
        mean_disp = 0 # 시차 평균을 0으로 처리합니다.
        mean_depth = 0 # 깊이 평균도 0으로 처리합니다.

    # 계산된 해당 영역의 시차 평균과 깊이 평균을 딕셔너리에 저장합니다.
    results[name] = {'disp': mean_disp, 'depth': mean_depth} 

# -----------------------------
# 4. 결과 출력 및 [요구사항: 결과 해석]
# -----------------------------
print("\n=== ROI별 평균 Disparity 및 Depth 측정 결과 ===") # 콘솔창에 결과 시작을 알리는 문구를 출력합니다.
closest_name = "" # 가장 거리가 가까운 영역의 이름을 저장할 변수입니다.
farthest_name = "" # 가장 거리가 먼 영역의 이름을 저장할 변수입니다.
min_depth = float('inf') # 최소 깊이를 찾기 위해 초기값을 무한대(inf)로 설정합니다.
max_depth = 0 # 최대 깊이를 찾기 위해 초기값을 0으로 설정합니다.

for name, res in results.items(): # 계산된 영역별 결과 딕셔너리를 하나씩 순회합니다.
    # 각 영역의 이름과 평균 시차, 평균 깊이를 콘솔에 출력합니다.
    print(f"{name:>8} - 평균 Disparity: {res['disp']:6.2f}, 평균 Depth: {res['depth']:.4f}")
    
    if res['depth'] > 0: # 깊이 평균이 0보다 큰 유효한 결과일 경우에만 거리 비교를 수행합니다.
        if res['depth'] < min_depth: # 현재 깊이가 저장된 최소 깊이보다 작다면 (더 가깝다면)
            min_depth = res['depth'] # 최소 깊이 값을 현재 값으로 갱신합니다.
            closest_name = name # 가장 가까운 영역의 이름을 현재 영역 이름으로 갱신합니다.
            
        if res['depth'] > max_depth: # 현재 깊이가 저장된 최대 깊이보다 크다면 (더 멀다면)
            max_depth = res['depth'] # 최대 깊이 값을 현재 값으로 갱신합니다.
            farthest_name = name # 가장 먼 영역의 이름을 현재 영역 이름으로 갱신합니다.

# 해석 출력
print("- 측정 결과   : 'Teddy' 영역의 Disparity가 가장 크고(Depth가 가장 작고), 'Painting' 영역의 Disparity가 가장 작습니다(Depth가 가장 큽니다).")
print(f"- 결론   : 따라서 세 ROI 중 카메라에서 가장 가까운 영역은 [{closest_name}] 이며, 가장 먼 영역은 [{farthest_name}] 입니다.") 

# -----------------------------
# 5. disparity 시각화 (컬러 매핑)
# -----------------------------
disp_tmp = disparity.copy() # 시각화 처리 중 원본 데이터가 손상되지 않도록 복사본을 만듭니다.
disp_tmp[disp_tmp <= 0] = np.nan # 0 이하의 매칭 실패 픽셀을 Not a Number(NaN)로 변환하여 계산에서 제외합니다.

if np.all(np.isnan(disp_tmp)): # 만약 배열의 모든 값이 NaN이라면 (모두 매칭 실패)
    raise ValueError("유효한 disparity 값이 없습니다.") # 에러를 발생시키고 중단합니다.

d_min = np.nanpercentile(disp_tmp, 5) # 노이즈를 피하기 위해 하위 5% 위치의 시차 값을 최솟값 기준으로 잡습니다.
d_max = np.nanpercentile(disp_tmp, 95) # 노이즈를 피하기 위해 상위 95% 위치의 시차 값을 최댓값 기준으로 잡습니다.

if d_max <= d_min: # 최댓값과 최솟값이 같아 분모가 0이 되는 것을 방지하기 위해
    d_max = d_min + 1e-6 # 최댓값에 아주 작은 수를 더해줍니다.

# 시차 값들을 0.0 에서 1.0 사이의 비율로 정규화(스케일링) 합니다.
disp_scaled = (disp_tmp - d_min) / (d_max - d_min) 

# 계산 과정에서 0 미만이나 1 초과로 벗어난 값들을 0과 1 사이로 강제로 잘라냅니다(clip).
disp_scaled = np.clip(disp_scaled, 0, 1) 

# 최종적으로 화면에 그릴 빈 8비트 이미지(0~255) 행렬을 생성합니다.
disp_vis = np.zeros_like(disparity, dtype=np.uint8) 

# NaN이 아닌 유효한 픽셀들의 위치(True, False 행렬)를 구합니다.
valid_disp = ~np.isnan(disp_tmp) 

# 0~1 사이의 정규화된 비율에 255를 곱해 0~255 사이의 픽셀 밝기값(정수)으로 변환하여 빈 이미지에 채워 넣습니다.
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8) 

# 흑백 이미지에 파란색(먼 곳)~빨간색(가까운 곳) 컬러맵(JET)을 덧입혀 보기 좋게 만듭니다.
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET) 

# -----------------------------
# 6. depth 시각화 (컬러 매핑)
# -----------------------------
# 시차 시각화와 마찬가지로 깊이를 표현할 빈 8비트 이미지 행렬을 생성합니다.
depth_vis = np.zeros_like(depth_map, dtype=np.uint8) 

if np.any(valid_mask): # 매칭이 성공한 픽셀이 화면에 존재한다면
    depth_valid = depth_map[valid_mask] # 유효한 픽셀 영역의 깊이 값들만 추출합니다.

    z_min = np.percentile(depth_valid, 5) # 하위 5% 위치의 깊이 값을 최소 기준으로 잡습니다.
    z_max = np.percentile(depth_valid, 95) # 상위 95% 위치의 깊이 값을 최대 기준으로 잡습니다.

    if z_max <= z_min: # 분모가 0이 되는 것을 방지합니다.
        z_max = z_min + 1e-6 

    # 깊이 값들을 0.0 에서 1.0 사이의 비율로 정규화합니다.
    depth_scaled = (depth_map - z_min) / (z_max - z_min) 
    
    # 0과 1 사이를 벗어난 값들을 잘라냅니다.
    depth_scaled = np.clip(depth_scaled, 0, 1) 

    # Depth는 값이 클수록(멀수록) 색상이 파란색으로, 작을수록(가까울수록) 빨간색으로 나와야 하므로 1.0에서 빼어 반전시킵니다.
    depth_scaled = 1.0 - depth_scaled 
    
    # 정규화된 값에 255를 곱해 이미지 픽셀 값으로 변환 후 할당합니다.
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8) 

# 계산된 깊이 이미지에 컬러맵(JET)을 적용합니다.
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) 

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
# 원본 이미지 위에 사각형을 그리기 위해 복사본을 생성합니다.
left_vis = left_color.copy() 
right_vis = right_color.copy() 

for name, (x, y, w, h) in rois.items(): # 설정된 3개의 관심 영역을 순회하며
    # 왼쪽 복사본 이미지의 해당 좌표에 초록색(0, 255, 0), 두께 2의 직사각형을 그립니다.
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    # 직사각형 바로 위쪽에 영역의 이름표(텍스트)를 초록색으로 덧붙입니다.
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 

    # 오른쪽 복사본 이미지에도 똑같이 초록색 직사각형을 그립니다.
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    # 오른쪽 복사본 이미지 직사각형 위에도 이름표를 덧붙입니다.
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) 

# -----------------------------
# 8. 파일 저장
# -----------------------------
# 'outputs' 폴더 안에 생성된 시차 맵 이미지를 'disparity.png'라는 이름으로 저장합니다.
cv2.imwrite(str(output_dir / 'disparity.png'), disparity_color) 
# 생성된 깊이 맵 이미지를 'depth.png'라는 이름으로 저장합니다.
cv2.imwrite(str(output_dir / 'depth.png'), depth_color) 
# 사각형 박스와 글씨가 그려진 왼쪽 이미지를 'left_roi.png'라는 이름으로 저장합니다.
cv2.imwrite(str(output_dir / 'left_roi.png'), left_vis) 

# -----------------------------
# 9. 화면 출력
# -----------------------------
# 실제 원본 왼쪽 이미지를 'Original' 창에 띄웁니다. (크기 조절 가능하도록 속성 추가)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.imshow('Original', left_color) 

# 만들어진 컬러 시차 맵(Disparity Map)을 창에 띄웁니다.
cv2.imshow('Disparity Map', disparity_color) 

# 만들어진 컬러 깊이 맵(Depth Map)을 창에 띄웁니다.
cv2.imshow('Depth Map', depth_color) 

# 사용자가 키보드의 아무 키나 누를 때까지 열려있는 창들을 닫지 않고 무한정 기다립니다.
cv2.waitKey(0) 

# 키보드 입력이 들어오면 모든 열려있는 팝업 창을 닫고 프로그램을 안전하게 종료합니다.
cv2.destroyAllWindows()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1340" height="593" alt="image" src="https://github.com/user-attachments/assets/f31acb0e-fd0d-4f3f-a778-e2a54fd102ed" />
<img width="1357" height="194" alt="image" src="https://github.com/user-attachments/assets/91a1e008-460a-42c4-ba62-2c2411ad87de" />
</div>
