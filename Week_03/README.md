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

<details><img width="2241" height="698" alt="image" src="https://github.com/user-attachments/assets/18188d26-6d35-4ca7-a5db-9a9046e84c3f" />

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
<img width="1498" height="590" alt="image" src="https://github.com/user-attachments/assets/7dbefb33-7554-46dc-963b-58bcdc69a02d" />
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
<img width="2241" height="698" alt="image" src="https://github.com/user-attachments/assets/49676f5d-8ea2-4a3e-87fc-c886cd5a7f1f" />
</div>
<br><br>

# 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출 (3.3.py)
[실습 목표] <br>
사용자가 지정한 사각형 영역을 초기 단서로 제공하여 GrabCut 알고리즘을 수행하고, 복잡한 이미지에서 전경(객체)과 배경을 분리해 내는 실습입니다.
<br><br>
[과제 설명] <br>
cv2.grabCut() 함수를 사용하여 사용자가 설정한 사각형 영역을 기반으로 대화식 분할을 수행했습니다. <br>
생성된 마스크 배열에서 배경과 관련된 값을 0으로, 전경과 관련된 값을 1로 수정한 뒤, np.where()를 이용해 이진화된 마스크 이미지를 만들고 이를 원본 이미지에 곱해 배경을 완벽히 제거했습니다. <br>
Matplotlib를 활용하여 원본 이미지, 생성된 이진 마스크 이미지, 그리고 객체만 추출된 최종 결과 이미지를 나란히 시각화했습니다. <br>

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# --- 전역 변수 설정 (마우스 상태 및 좌표 저장용) ---
drawing = False # 마우스 클릭 여부를 추적하는 플래그
ix, iy = -1, -1 # 드래그 시작점 (x, y) 좌표
rect = (0, 0, 0, 0) # GrabCut에 전달할 최종 (x, y, w, h)
img_disp = None # 화면 출력용 이미지 복사본
img_grab = None # 원본 이미지

# --- 마우스 이벤트 처리 콜백 함수 ---
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, rect, img_disp, img_grab

    # 1) 마우스 왼쪽 버튼을 누르는 순간: 드래그 시작점 기록
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    # 2) 마우스를 누른 채 이동하는 중: 사각형을 실시간으로 화면에 그림
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 잔상이 남지 않도록 원본 복사본을 매번 새로 불러와 그 위에 사각형을 덧그립니다.
            img_disp = img_grab.copy()
            cv.rectangle(img_disp, (ix, iy), (x, y), (0, 255, 0), 2)

    # 3) 마우스 왼쪽 버튼을 떼는 순간: 드래그 종료 및 최종 사각형 좌표 확정
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img_disp, (ix, iy), (x, y), (0, 255, 0), 2)
        
        # 시작점과 끝점 중 어디가 더 작은지 비교하여 (x, y, width, height) 형태를 안전하게 계산합니다.
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)
        rect = (x1, y1, x2 - x1, y2 - y1)

# --- 1. 원본 이미지 불러오기 ---
img_grab = cv.imread('coffee_cup.jpg')

if img_grab is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인해 주세요.")
else:
    # --- 2. 직접 구현한 마우스 드래그를 이용한 ROI 설정 ---
    img_disp = img_grab.copy()
    
    # 창을 먼저 생성하고 콜백 함수를 연결해야 정상적으로 마우스 입력을 감지할 수 있습니다.
    cv.namedWindow('Draw ROI (Press ENTER to confirm)')
    cv.setMouseCallback('Draw ROI (Press ENTER to confirm)', draw_roi)

    print("■ 팝업 창에서 커피잔 주변을 마우스로 드래그하여 사각형을 그리세요.")
    print("■ 사각형을 그린 후 [Enter] 키를 누르면 GrabCut 연산이 시작됩니다.")

    while True:
        cv.imshow('Draw ROI (Press ENTER to confirm)', img_disp)
        k = cv.waitKey(1) & 0xFF
        # Enter 키(13)를 누르면 무한 루프를 탈출합니다.
        if k == 13: 
            break
            
    cv.destroyAllWindows()

    # 사각형 크기가 0이거나 잘못 그려진 경우를 대비한 최소한의 예외 처리
    if rect[2] <= 0 or rect[3] <= 0:
        print("[경고] 영역이 제대로 지정되지 않아 기본 좌표를 사용합니다.")
        h, w = img_grab.shape[:2]
        rect = (int(w * 0.15), int(h * 0.15), int(w * 0.70), int(h * 0.70))

    # --- 3. GrabCut 알고리즘 초기화 ---
    mask = np.zeros(img_grab.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # --- 4. GrabCut 대화식 분할 수행 ---
    # 직접 드래그하여 확정된 rect 좌표를 활용하여 연산합니다.
    cv.grabCut(img_grab, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    # --- 5. 마스크 값 후처리 및 배경 제거 ---
    mask_binary = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
    extracted_img = img_grab * mask_binary[:, :, np.newaxis]

    # --- 6. Matplotlib를 사용한 결과 시각화 ---
    plt.figure(figsize=(15, 5))

    # 첫 번째: 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(img_grab, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # 두 번째: 객체 추출 결과를 마스크 형태로 시각화
    plt.subplot(1, 3, 2)
    plt.imshow(mask_binary * 255, cmap='gray')
    plt.title('GrabCut Mask')
    plt.axis('off')

    # 세 번째: 배경이 제거된 이미지
    plt.subplot(1, 3, 3)
    plt.imshow(cv.cvtColor(extracted_img, cv.COLOR_BGR2RGB))
    plt.title('Extracted Object')
    plt.axis('off')

    plt.tight_layout()
    plt.show()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="2241" height="721" alt="image" src="https://github.com/user-attachments/assets/c08c8829-f78e-4d6b-9d3e-f6470f4ff23c" />
</div>
