<div align="center">

# 컴퓨터 비전 3주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>
에지 검출(소벨, 캐니), 허프 변환을 이용한 직선 검출, <br>
그리고 GrabCut을 이용한 대화식 영역 분할 및 객체 추출을 수행하는 실습입니다. <br>

</div>
<br><br>

# 1. 소벨 에지 검출 및 결과 시각화 (3.1.py)
[실습 목표]<br>
이미지를 그레이스케일로 변환하고 Sobel 필터를 사용하여 x축과 y축 방향의 에지를 검출한 후, 검출된 에지 강도 이미지를 시각화하는 실습입니다.
<br><br>
[과제 설명]<br>
cv2.imread()를 사용하여 이미지를 불러오고 cv2.cvtColor()를 통해 그레이스케일로 변환했습니다. <br>
cv2.Sobel() 함수를 사용하여 x축과 y축 방향의 에지를 검출하고, cv2.magnitude()를 사용해 최종적인 에지 강도를 계산했습니다. <br>
계산된 에지 강도 이미지를 cv2.convertScaleAbs()로 변환한 뒤, Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 흑백으로 나란히 시각화했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv # 영상 처리를 위한 OpenCV 라이브러리를 가져옵니다.
import numpy as np # 행렬 및 다차원 배열 연산을 위한 NumPy 라이브러리를 가져옵니다.
import matplotlib.pyplot as plt # 결과를 화면에 나란히 시각화하기 위해 Matplotlib의 pyplot 모듈을 가져옵니다.

# 1. 이미지 불러오기 및 그레이스케일 변환
# 'edgeDetectionImage.jpg' 파일을 BGR(컬러) 포맷 행렬로 읽어와 img_sobel 변수에 할당합니다.
img_sobel = cv.imread('edgeDetectionImage.jpg') 

# 파일 경로가 틀렸거나, 파일이 손상되어 이미지를 정상적으로 메모리에 적재하지 못했는지 검증합니다.
if img_sobel is None:
    # 이미지를 찾지 못했을 경우 콘솔에 에러 메시지를 출력하고 이후 연산을 진행하지 않습니다.
    print("이미지를 찾을 수 없습니다.")
else:
    # 에지 검출은 '밝기의 변화량(명암차)'을 미분하여 구하므로, 색상 정보(Color)는 노이즈가 될 수 있습니다. 
    # 따라서 BGR 컬러 이미지를 단일 채널인 흑백(Grayscale) 이미지로 변환합니다.
    gray_sobel = cv.cvtColor(img_sobel, cv.COLOR_BGR2GRAY)

    # 2. x축, y축 방향 소벨 에지 검출
    # [중요] cv.CV_64F를 사용하는 이유: 밝기 변화량(미분값)은 음수가 될 수도 있습니다(밝은 곳 -> 어두운 곳). 
    # 일반적인 8비트 부호 없는 정수(uint8)를 쓰면 음수 값이 모두 0으로 잘려(Clipping) 정보가 손실되므로, 
    # 음수와 소수점 표현이 가능한 64비트 실수형(Float)으로 출력 데이터 타입을 지정하는 것입니다.
    
    # x축 방향(가로 방향)으로 명암이 변하는 수직선 에지를 검출합니다. (dx=1, dy=0, 커널사이즈=3x3)
    grad_x = cv.Sobel(gray_sobel, cv.CV_64F, 1, 0, ksize=3)
    
    # y축 방향(세로 방향)으로 명암이 변하는 수평선 에지를 검출합니다. (dx=0, dy=1, 커널사이즈=3x3)
    grad_y = cv.Sobel(gray_sobel, cv.CV_64F, 0, 1, ksize=3)

    # 3. 에지 강도 계산 및 형변환
    # 피타고라스 정리(sqrt(x^2 + y^2))를 이용해 x축 미분값과 y축 미분값을 합산하여 최종적인 에지의 강도(Magnitude)를 구합니다.
    magnitude = cv.magnitude(grad_x, grad_y)
    
    # 계산된 magnitude는 여전히 실수형(64F)이므로 화면에 픽셀로 표시할 수 없습니다.
    # 절댓값을 취한 뒤 화면 출력용 기본 타입인 8비트 부호 없는 정수(uint8, 0~255)로 형변환합니다.
    magnitude_uint8 = cv.convertScaleAbs(magnitude)

    # 4. Matplotlib로 시각화 (원본과 에지 강도 이미지)
    # 가로 10인치, 세로 5인치 크기의 새로운 그림(Figure) 창을 생성합니다.
    plt.figure(figsize=(10, 5))
    
    # 1행 2열로 화면을 분할하고, 그 중 첫 번째(왼쪽) 영역을 선택합니다.
    plt.subplot(1, 2, 1)
    # OpenCV는 이미지를 BGR 순서로 읽지만, Matplotlib는 RGB 순서로 화면에 그립니다.
    # 따라서 색상이 뒤집히지 않도록 BGR을 RGB로 변환하여 출력합니다.
    plt.imshow(cv.cvtColor(img_sobel, cv.COLOR_BGR2RGB))
    # 왼쪽 그림의 상단에 'Original Image'라는 제목을 답니다.
    plt.title('Original Image')
    # 이미지 주변의 불필요한 x축, y축 눈금과 테두리를 숨깁니다.
    plt.axis('off')

    # 1행 2열로 분할된 화면 중 두 번째(오른쪽) 영역을 선택합니다.
    plt.subplot(1, 2, 2)
    # 형변환이 완료된 에지 강도 이미지(magnitude_uint8)를 띄웁니다.
    # 단일 채널(흑백) 이미지이므로 cmap='gray'를 명시해야 녹색/노란색 등으로 왜곡되어 보이지 않습니다.
    plt.imshow(magnitude_uint8, cmap='gray')
    # 오른쪽 그림의 상단에 'Sobel Edge Magnitude'라는 제목을 답니다.
    plt.title('Sobel Edge Magnitude')
    # 오른쪽 그림 역시 축 눈금을 숨깁니다.
    plt.axis('off')

    # 지금까지 설정한 모든 subplot들을 실제 윈도우 창으로 렌더링하여 화면에 띄웁니다.
    plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1909" height="760" alt="image" src="https://github.com/user-attachments/assets/55fc9b8b-4154-4685-bca8-f448c1a57d3c" />
<img width="724" height="182" alt="image" src="https://github.com/user-attachments/assets/b31a2dac-d9b5-48b1-9803-f4aced5057c7" />
</div>
<br><br>

# 2. 캐니 에지 및 허프 변환을 이용한 직선 검출 (3.2.py)
[실습 목표] <br>
이미지에 캐니 에지 검출을 사용하여 에지 맵을 생성하고, 허프 변환을 사용하여 이미지에서 직선을 찾아 원본에 표시하는 실습입니다.
<br><br>
[과제 설명] <br>
cv2.Canny() 함수에 이력 임계값(100, 200)을 적용하여 노이즈가 제거된 에지 맵을 생성했습니다. <br>
cv2.HoughLinesP()를 사용하여 끊어진 에지 사이에서 직선 성분을 검출하고, cv2.line()을 사용하여 검출된 직선을 원본 이미지 위에 빨간색 선으로 그렸습니다. <br>
마지막으로 Matplotlib를 사용하여 원본 이미지와 직선이 그려진 결과 이미지를 나란히 시각화하여 확인했습니다. <br>
 
<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv # 컴퓨터 비전 알고리즘 연산을 위한 OpenCV 라이브러리입니다.
import numpy as np # 허프 변환 각도(라디안) 계산 등 수학적 처리를 위한 NumPy 라이브러리입니다.
import matplotlib.pyplot as plt # 결과를 화면에 시각화하기 위한 라이브러리입니다.

# 'dabo.jpg' 이미지를 메모리에 컬러(BGR) 배열로 적재합니다.
img_hough = cv.imread('dabo.jpg')

# 파일 입출력 예외 처리: 경로 오류나 파일 손상 여부를 검증합니다.
if img_hough is None:
    print("이미지를 찾을 수 없습니다.")
else:
    # 에지 검출은 '명암의 급격한 변화'를 찾는 과정이므로, 컬러(BGR) 연산은 불필요한 연산량 증가와 노이즈를 유발합니다.
    # 따라서 단일 채널인 흑백(Grayscale) 이미지로 변환합니다.
    gray_hough = cv.cvtColor(img_hough, cv.COLOR_BGR2GRAY)

    # 1. 캐니 에지 검출 (Canny Edge Detection)
    # 캐니 알고리즘은 노이즈 제거 -> 소벨 미분 -> 비최대 억제(Non-Maximum Suppression) -> 이력 임계값(Hysteresis Thresholding)의 4단계를 거칩니다.
    # 100(T_low): 하위 임계값. 이 값보다 약한 에지는 모두 버립니다.
    # 200(T_high): 상위 임계값. 이 값보다 강한 에지는 '확실한 에지(Strong Edge)'로 간주합니다.
    # 100~200 사이의 값(Weak Edge)은 확실한 에지와 연결되어 있을 때만 에지로 인정하여 선이 끊어지는 것을 방지합니다.
    edges = cv.Canny(gray_hough, 100, 200)

    # 2. 허프 변환으로 직선 검출 (Probabilistic Hough Line Transform)
    # [주의] 이 함수는 단순한 선 긋기가 아닙니다. 에지 픽셀들을 (rho, theta) 파라미터 공간으로 보내 '투표(Voting)'를 시키는 수학적 과정입니다.
    lines = cv.HoughLinesP(
        edges,            # 입력 이미지: 반드시 흑백 1채널의 '에지 맵(Canny 결과물)'을 넣어야 합니다.
        1,                # rho (거리 해상도): 원점에서 직선까지의 수직 거리를 1픽셀 단위로 촘촘하게 탐색하겠다는 의미입니다.
        np.pi / 180,      # theta (각도 해상도): 각도를 1도(pi/180 라디안) 단위로 세밀하게 탐색하겠다는 의미입니다.
        threshold=50,     # 임계값(투표수): 파라미터 공간(Hough Space)에서 최소 50표 이상 겹친(교차한) 점들만 '유의미한 직선'으로 인정합니다.
        minLineLength=50, # 최소 선 길이: 검출된 직선의 길이가 최소 50픽셀 이상이어야만 결과로 반환합니다. (자잘한 노이즈 선 제거)
        maxLineGap=10     # 최대 선 간격: 에지 픽셀이 끊겨 있더라도, 그 간격이 10픽셀 이하라면 하나의 이어진 직선으로 간주합니다.
    )

    # 3. 원본 이미지에 빨간색 직선 그리기
    # 검출된 선을 그릴 도화지 역할을 할 원본 이미지의 복사본을 생성합니다. (원본 데이터 훼손 방지)
    img_hough_drawn = img_hough.copy()
    
    # 허프 변환 결과, 조건(threshold 등)을 만족하는 직선이 단 하나도 없을 경우 에러가 나는 것을 방지합니다.
    if lines is not None:
        # 반환된 lines 배열은 검출된 선분들의 양 끝점 좌표 뭉치입니다. 이를 하나씩 꺼냅니다.
        for line in lines:
            # line[0]에 들어있는 시작점(x1, y1)과 끝점(x2, y2) 좌표를 언패킹(Unpacking)합니다.
            x1, y1, x2, y2 = line[0]
            # cv.line 함수를 이용해 계산된 두 점을 잇는 선분을 그립니다.
            # (0, 0, 255): OpenCV는 BGR 순서를 따르므로 Red 채널만 255로 켠 빨간색을 의미합니다. 2는 선의 두께(픽셀)입니다.
            cv.line(img_hough_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 4. Matplotlib로 시각화
    # 가로 12인치, 세로 5인치 비율의 Figure(그림 창)를 메모리에 생성합니다.
    plt.figure(figsize=(12, 5))
    
    # 1행 2열로 화면을 분할하고, 왼쪽(첫 번째) 영역을 지정합니다.
    plt.subplot(1, 2, 1)
    # Matplotlib는 RGB를 표준으로 사용하므로, OpenCV로 읽은 BGR 이미지를 RGB로 변환하여 색상 왜곡을 막습니다.
    plt.imshow(cv.cvtColor(img_hough, cv.COLOR_BGR2RGB))
    plt.title('Original Image') # 제목 설정
    plt.axis('off')             # x, y축의 눈금(Tick)과 테두리를 숨겨 이미지만 깔끔하게 보여줍니다.

    # 분할된 화면 중 오른쪽(두 번째) 영역을 지정합니다.
    plt.subplot(1, 2, 2)
    # 직선이 그려진 결과 이미지 역시 BGR에서 RGB로 변환하여 출력합니다.
    plt.imshow(cv.cvtColor(img_hough_drawn, cv.COLOR_BGR2RGB))
    plt.title('Hough Lines Detected')
    plt.axis('off')

    # 메모리에 구성된 두 개의 Subplot을 실제 화면(윈도우)에 렌더링하여 표시합니다.
    plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1178" height="427" alt="image" src="https://github.com/user-attachments/assets/5604392a-f5b1-464f-9f05-c4499ce0cd94" />
</div>
<br><br>

# 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출 (3.3.py)
[실습 목표] <br>
사용자가 지정한 사각형 영역을 초기 단서로 제공하여 GrabCut 알고리즘을 수행하고, 복잡한 이미지에서 전경(객체)과 배경을 분리해 내는 실습입니다.
<br><br>
[과제 설명] <br>
cv2.grabCut() 함수를 사용하여 사용자가 설정한 사각형 영역(rect)을 기반으로 대화식 분할을 수행했습니다. <br>
생성된 마스크 배열에서 배경과 관련된 값을 0으로, 전경과 관련된 값을 1로 수정한 뒤, np.where()를 이용해 이진화된 마스크 이미지를 만들고 이를 원본 이미지에 곱해 배경을 완벽히 제거했습니다. <br>
Matplotlib를 활용하여 원본 이미지, 생성된 이진 마스크 이미지, 그리고 객체만 추출된 최종 결과 이미지를 나란히 시각화했습니다. <br>

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
