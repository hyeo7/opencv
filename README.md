<div align="center">

컴퓨터 비전 1주차 실습

이번 실습은 Python과 OpenCV 라이브러리를 활용하여 이미지를 다루고,
마우스와 키보드 이벤트를 통해 이미지 위에 그림을 그리거나
특정 영역을 잘라내어 저장하는 실습입니다.

</div>

1. 이미지 컬러/흑백 나란히 출력하기 (1.1.1.py)

[실습 목표]
하나의 이미지를 컬러와 흑백 두 가지 버전으로 불러온 뒤, 화면 비율을 유지하면서 크기를 조절하고 두 이미지를 가로로 나란히 이어 붙여 한 화면에 출력하는 실습입니다.

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 # OpenCV 라이브러리를 불러옵니다. 이미지 및 영상 처리에 사용됩니다.
import numpy as np # 행렬 및 다차원 배열 처리를 위한 numpy 라이브러리를 불러옵니다.

# 1. 이미지 불러오기
# 'soccer.jpg' 파일을 컬러 이미지(기본값)로 읽어옵니다.
img_color = cv2.imread('soccer.jpg')
# 'soccer.jpg' 파일을 흑백(그레이스케일) 이미지로 읽어옵니다.
img_gray = cv2.imread('soccer.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 이미지 크기 조절 (비율 유지)
width = 500 # 변경할 목표 가로 길이를 500 픽셀로 설정합니다.
# 원본 이미지의 비율을 유지하기 위해 세로 길이를 비례식으로 계산합니다. (정수형으로 변환)
height = int(img_color.shape[0] * (width / img_color.shape[1]))

# 계산된 너비와 높이로 컬러 이미지의 크기를 조절합니다.
img_color_resized = cv2.resize(img_color, (width, height))
# 계산된 너비와 높이로 흑백 이미지의 크기를 조절합니다.
img_gray_resized = cv2.resize(img_gray, (width, height))

# 3. 흑백 이미지를 3차원으로 변환 
# 흑백 이미지는 1차원이므로, 3차원인 컬러 이미지와 나란히 붙이려면 형태를 3차원인 BGR로 맞춰야 합니다. 
img_gray_3ch = cv2.cvtColor(img_gray_resized, cv2.COLOR_GRAY2BGR)

# 4. 이미지 가로로 나란히 붙이기
# numpy의 hstack(horizontal stack) 함수를 사용하여 컬러 이미지와 3차원 흑백 이미지를 가로로 이어 붙입니다.
result = np.hstack((img_color_resized, img_gray_3ch))

# 5. 결과 창에 출력
# 'Color and Grayscale'이라는 이름의 창을 띄우고 결과 이미지를 보여줍니다.
cv2.imshow('Color and Grayscale', result)

# 키보드 입력이 있을 때까지 창을 무한정 대기시킵니다. (0은 무한 대기를 의미)
cv2.waitKey(0)

# 사용자의 입력이 들어오면 생성된 모든 OpenCV 창을 닫고 메모리를 해제합니다.
cv2.destroyAllWindows()```
```


</div>
</details>

실행 결과 화면

<div align="center">
<img width="1493" height="535" alt="실행 결과 1" src="https://github.com/user-attachments/assets/68c8514b-fc8d-486a-b531-e70310b59743" />
</div>

2. 마우스 이벤트로 그림 그리기 (1.1.2.py)

[실습 목표]
불러온 이미지 위에 마우스 클릭과 드래그를 이용하여 그림을 그리는 미니 그림판 프로그램입니다. 좌클릭은 파란색, 우클릭은 빨간색으로 그려지며, 키보드 +와 - 키를 이용해 붓의 크기를 조절할 수 있습니다.

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv # OpenCV 라이브러리를 불러오고, 코드 작성을 편하게 하기 위해 'cv'라는 별칭으로 지정합니다.

# 1. 축구 이미지 불러오기
# 'soccer.jpg' 파일을 읽어와 img 변수에 저장합니다. 그림을 그릴 바탕(도화지) 역할입니다.
img = cv.imread('soccer.jpg') 

# 이미지가 제대로 불러와졌는지 예외 처리를 통해 확인합니다.
if img is None:
    print("이미지를 불러올 수 없습니다. 파일명과 경로를 확인해 주세요.")
    exit() # 이미지를 찾을 수 없다면 프로그램을 안전하게 종료합니다.

# 초기 붓 크기(원의 반지름)는 5 픽셀로 설정합니다.
brush_size = 5
# 마우스 왼쪽 버튼이 눌려있는지(드래그 상태) 기억하는 상태 변수입니다.
drawing_left = False
# 마우스 오른쪽 버튼이 눌려있는지 기억하는 상태 변수입니다.
drawing_right = False

# 마우스 이벤트를 처리할 콜백 함수를 정의합니다.
def paint(event, x, y, flags, param):
    # 전역 변수들을 함수 내부에서 값을 수정하며 사용할 수 있도록 global로 선언합니다.
    global brush_size, drawing_left, drawing_right, img

    # 좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기 
    # 마우스 왼쪽 버튼을 누르는 순간 발생하는 이벤트
    if event == cv.EVENT_LBUTTONDOWN:
        drawing_left = True # 왼쪽 그리기 모드를 켭니다.
        # 클릭한 좌표(x, y)에 파란색(255, 0, 0), 지정된 붓 크기로 속이 꽉 찬(-1) 원을 그립니다.
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1) 

    # 마우스 오른쪽 버튼을 누르는 순간 발생하는 이벤트
    elif event == cv.EVENT_RBUTTONDOWN:
        drawing_right = True # 오른쪽 그리기 모드를 켭니다.
        # 클릭한 좌표(x, y)에 빨간색(0, 0, 255)으로 속이 꽉 찬 원을 그립니다.
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1) 

    # 마우스가 움직일 때마다 발생하는 이벤트 (드래그 처리)
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing_left: # 왼쪽 버튼을 누른 채로 움직이면 파란색 원을 연속해서 그립니다.
            cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
        elif drawing_right: # 오른쪽 버튼을 누른 채로 움직이면 빨간색 원을 연속해서 그립니다.
            cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)

    # 마우스 왼쪽 버튼에서 손을 뗄 때 발생하는 이벤트
    elif event == cv.EVENT_LBUTTONUP:
        drawing_left = False # 왼쪽 그리기 모드를 끕니다.
        
    # 마우스 오른쪽 버튼에서 손을 뗄 때 발생하는 이벤트
    elif event == cv.EVENT_RBUTTONUP:
        drawing_right = False # 오른쪽 그리기 모드를 끕니다.

# 'Paint'라는 이름의 새로운 윈도우 창을 메모리에 만듭니다.
cv.namedWindow('Paint')
# 'Paint' 창에서 일어나는 모든 마우스 조작을 우리가 만든 paint 함수로 연결합니다.
cv.setMouseCallback('Paint', paint)

# 그림을 그리는 동안 화면을 실시간으로 업데이트하기 위해 무한 루프를 돕니다.
while True:
    # 그림이 추가된 img를 'Paint' 창에 띄워 화면에 보여줍니다.
    cv.imshow('Paint', img)
    
    # 키보드 입력을 대기합니다.
    key = cv.waitKey(1) & 0xFF

    # 키보드 'q' 키를 누르면 프로그램을 종료합니다.
    if key == ord('q'):
        break
        
    # 키보드 '+' 기호 입력 시 붓 크기를 증가시킵니다.
    elif key == ord('+') or key == ord('='):
        brush_size = min(15, brush_size + 1)
        print(f"현재 붓 크기: {brush_size}")
        
    # 키보드 '-' 기호 입력 시 붓 크기를 감소시킵니다.
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)
        print(f"현재 붓 크기: {brush_size}")

cv.destroyAllWindows()```
```


</div>
</details>

실행 결과 화면

<div align="center">
<img width="2148" height="1354" alt="실행 결과 2-1" src="https://github.com/user-attachments/assets/75ead328-b07a-46a4-840c-09975337aff9" />







<img width="1725" height="914" alt="실행 결과 2-2" src="https://github.com/user-attachments/assets/ad539da6-dbf3-43ff-ad36-d93e3c086700" />
</div>

3. 마우스 드래그로 영역(ROI) 선택 및 저장하기 (1.1.3.py)

[실습 목표]
이미지 위에서 마우스를 드래그하여 원하는 영역(ROI: Region of Interest)을 사각형으로 선택하고, 해당 부분을 잘라내어 별도의 창에 띄우는 프로그램입니다. 키보드를 조작하여 선택 영역을 초기화하거나 이미지 파일로 저장할 수 있습니다.

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv # OpenCV 라이브러리를 'cv'라는 별칭으로 불러옵니다.

# 1. 이미지를 불러오고 화면에 출력하기 위한 준비
img = cv.imread('soccer.jpg')

if img is None:
    print("축구 이미지를 불러올 수 없습니다. 파일명과 경로를 확인해 주세요.")
    exit()

clone = img.copy() # 원본 이미지 복사본 생성
roi = None
drawing = False
start_pt = (-1, -1)

# 2. 마우스 이벤트를 처리하는 콜백 함수 정의
def select_roi(event, x, y, flags, param):
    global start_pt, drawing, clone, img, roi

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            clone = img.copy() 
            cv.rectangle(clone, start_pt, (x, y), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        clone = img.copy()
        cv.rectangle(clone, start_pt, (x, y), (0, 255, 0), 2)

        x1, y1 = min(start_pt[0], x), min(start_pt[1], y)
        x2, y2 = max(start_pt[0], x), max(start_pt[1], y)

        if x2 > x1 and y2 > y1:
            roi = img[y1:y2, x1:x2]
            cv.imshow("ROI", roi)

cv.namedWindow('Image')
cv.setMouseCallback('Image', select_roi)

while True:
    cv.imshow('Image', clone)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):
        clone = img.copy()
        if cv.getWindowProperty("ROI", cv.WND_PROP_VISIBLE) >= 1:
            cv.destroyWindow("ROI")
        roi = None
        print("영역 선택이 초기화되었습니다. 다시 드래그해 주세요.")

    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('saved_roi.jpg', roi)
            print("선택한 영역이 'saved_roi.jpg'로 성공적으로 저장되었습니다.")
        else:
            print("저장할 영역이 아직 선택되지 않았습니다.")

    elif key == ord('q'):
        break

cv.destroyAllWindows()```
```


</div>
</details>

실행 결과 화면

<div align="center">
<img width="2147" height="1241" alt="실행 결과 3" src="https://github.com/user-attachments/assets/e5e29e60-d59a-425d-bc90-b16243d9a918" />
</div>
