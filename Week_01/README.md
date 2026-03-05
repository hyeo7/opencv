<div align="center">

# 컴퓨터 비전 1주차 실습
이번 실습은 Python과 OpenCV 라이브러리를 활용하여 <br>이미지를 다루고,
마우스와 키보드 이벤트를 통해 이미지 위에 그림을 그리거나
특정 영역을 잘라내어 저장합니다.

</div>
<br><br>

# 1. 이미지 불러오기 및 그레이스케일 변환 (1.1.1.py) 
[실습 목표]<br>
하나의 이미지를 컬러와 흑백 두 가지 버전으로 불러온 뒤, 화면 비율을 유지하면서 크기를 조절하고 <br>두 이미지를 가로로 나란히 이어 붙여 한 화면에 출력하는 실습입니다.
<br><br>
[과제 설명]<br>
OpenCV를 사용하여 축구 이미지를 불러오고 화면에 출력하는 기본 실습입니다.<br>
불러온 원본 이미지를 cv.cvtColor() 함수를 이용해 흑백(그레이스케일)으로 변환했습니다.<br>
np.hstack()을 활용하여 원본 이미지와 그레이스케일 이미지를 가로로 나란히 연결해 한눈에 비교할 수 있도록 구현했습니다.

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 # OpenCV 라이브러리를 불러옵니다. 이미지 및 영상 처리에 사용됩니다.
import numpy as np # 행렬 및 다차원 배열 처리를 위한 numpy 라이브러리를 불러옵니다.

# 1. 이미지 불러오기
# 'soccer.jpg' 파일을 컬러 이미지로 읽어옵니다.
img_color = cv2.imread('soccer.jpg')
# 'soccer.jpg' 파일을 흑백 이미지로 읽어옵니다.
img_gray = cv2.imread('soccer.jpg', cv2.IMREAD_GRAYSCALE)

# 2. 이미지 크기
width = 500 # 변경할 목표 가로 길이를 500 픽셀로 설정합니다.
# 원본 이미지의 비율을 유지하기 위해 세로 길이를 비례식으로 계산합니다.
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

# 키보드 입력이 있을 때까지 창을 무한정 대기시킵니다. (0은 무한 대기)
cv2.waitKey(0)

# 사용자의 입력이 들어오면 생성된 모든 OpenCV 창을 닫고 메모리를 해제합니다.
cv2.destroyAllWindows()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="1493" height="535" alt="실행 결과 1" src="https://github.com/user-attachments/assets/68c8514b-fc8d-486a-b531-e70310b59743" />
</div>
<br><br>

# 2. 마우스 이벤트로 그림 그리기 (1.1.2.py)
[실습 목표] <br>
불러온 이미지 위에 마우스 클릭과 드래그를 이용하여 그림을 그리는 미니 그림판 프로그램입니다. 좌클릭은 파란색, 우클릭은 빨간색으로 그려지며, <br>
키보드 +와 - 키를 이용해 붓의 크기를 조절할 수 있습니다.
<br><br>
[과제 설명] <br>
마우스 입력 이벤트를 처리하여 축구 이미지 위에 직접 선을 그릴 수 있는 인터랙티브 페인팅 기능입니다. <br>
좌클릭 시 파란색, 우클릭 시 빨간색 원이 그려지며, 마우스 드래그를 통해 연속적인 그리기가 가능합니다. <br>
키보드의 '+', '-' 키 입력을 받아 실시간으로 붓 크기를 조절하되, 최소 1에서 최대 15 사이를 유지하도록 제한 조건을 두었습니다. <br>
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

[실행 결과 화면]
<div align="center">
<img width="2148" height="1354" alt="실행 결과 2-1" src="https://github.com/user-attachments/assets/75ead328-b07a-46a4-840c-09975337aff9" />
<img width="1725" height="914" alt="실행 결과 2-2" src="https://github.com/user-attachments/assets/ad539da6-dbf3-43ff-ad36-d93e3c086700" />
</div>
<br><br>

# 3. 마우스 드래그로 영역(ROI) 선택 및 저장하기 (1.1.3.py)
[실습 목표] <br>
이미지 위에서 마우스를 드래그하여 원하는 영역(ROI: Region of Interest)을 사각형으로 선택하고, 해당 부분을 잘라내어 별도의 창에 띄우는 프로그램입니다. 키보드를 조작하여 선택 영역을 초기화하거나 이미지 파일로 저장할 수 있습니다. 
<br><br>
[과제 설명] <br>
이미지 위에서 마우스를 클릭하고 드래그하여 사용자가 원하는 관심영역(ROI)을 초록색 사각형으로 지정합니다.  <br>
마우스를 놓으면 지정된 영역의 좌표를 numpy 슬라이싱으로 계산하여 해당 부분만 잘라내 별도의 창에 띄워줍니다.  <br>
키보드 입력을 통해 선택 영역을 다시 리셋('r' 키)하거나, 추출된 이미지를 파일로 저장('s' 키)하는 편의 기능을 추가했습니다.  <br>

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2 as cv # OpenCV 라이브러리입니다.

# 1. 이미지를 불러오고 화면에 출력하기 위한 준비
img = cv.imread('soccer.jpg') # 'soccer.jpg' 파일을 읽어서 변수 img에 저장합니다.

if img is None: # 이미지가 파일 경로에 없거나 불러오기에 실패했을 경우를 체크합니다.
    print("축구 이미지를 불러올 수 없습니다. 파일명과 경로를 확인해 주세요.") # 에러 메시지를 출력합니다.
    exit() # 프로그램의 실행을 강제로 종료합니다.

clone = img.copy() # 원본 이미지 위에 사각형을 그리면 원본이 손상되므로, 화면 출력용 복사본을 만듭니다.
roi = None # 나중에 잘라낸 이미지 영역을 담을 변수를 미리 비워둔 상태로 선언합니다.
drawing = False # 마우스 드래그 중인지(버튼이 눌려있는지) 확인하기 위한 상태 변수입니다.
start_pt = (-1, -1) # 드래그를 시작한 좌표(x, y)를 저장할 변수로, 처음에는 유효하지 않은 값으로 설정합니다.

# 2. 마우스 이벤트를 처리하는 콜백 함수 정의
def select_roi(event, x, y, flags, param): # 마우스 조작 시 호출되는 함수로, 이벤트 종류와 좌표 정보를 받습니다.
    global start_pt, drawing, clone, img, roi # 함수 외부에서 선언된 전역 변수들을 함수 내에서 수정하기 위해 선언합니다.

    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼을 처음 누르는 순간 실행됩니다.
        drawing = True # 이제 드래그가 시작되었음을 알리는 상태로 바꿉니다.
        start_pt = (x, y) # 현재 마우스가 위치한 곳의 x, y 좌표를 시작점으로 기록합니다.

    elif event == cv.EVENT_MOUSEMOVE: # 마우스가 화면 위에서 움직일 때마다 실행됩니다.
        if drawing: # 만약 왼쪽 버튼을 누른 채로 움직이고 있다면(드래그 중이라면)
            clone = img.copy() # 이전 사각형 잔상을 지우기 위해 원본 이미지로부터 다시 깨끗한 복사본을 가져옵니다.
            cv.rectangle(clone, start_pt, (x, y), (0, 255, 0), 2) # 시작점부터 현재 좌표까지 초록색(0, 255, 0) 사각형을 그립니다.

    elif event == cv.EVENT_LBUTTONUP: # 마우스 왼쪽 버튼에서 손을 떼는 순간 실행됩니다.
        drawing = False # 드래그가 끝났으므로 상태 변수를 거짓으로 바꿉니다.
        clone = img.copy() # 화면에 최종 선택된 사각형을 고정시키기 위해 원본을 다시 복사합니다.
        cv.rectangle(clone, start_pt, (x, y), (0, 255, 0), 2) # 최종적으로 결정된 영역에 사각형을 한 번 더 그려줍니다.

        x1, y1 = min(start_pt[0], x), min(start_pt[1], y) # 마우스를 어느 방향으로 끌어도 자르기가 가능하도록 최소 x, y 좌표를 구합니다.
        x2, y2 = max(start_pt[0], x), max(start_pt[1], y) # 마우스를 어느 방향으로 끌어도 자르기가 가능하도록 최대 x, y 좌표를 구합니다.

        if x2 > x1 and y2 > y1: # 드래그한 영역의 넓이가 존재할 경우에만 자르기를 수행합니다.
            roi = img[y1:y2, x1:x2] # 넘파이 슬라이싱을 사용하여 원본 이미지에서 해당 좌표만큼의 행렬 데이터를 추출합니다.
            cv.imshow("ROI", roi) # 잘라낸 이미지(ROI)를 'ROI'라는 이름의 새로운 창에 띄워 보여줍니다.

cv.namedWindow('Image') # 'Image'라는 이름의 윈도우 창을 미리 생성합니다.
cv.setMouseCallback('Image', select_roi) # 'Image' 창에서 발생하는 모든 마우스 동작을 select_roi 함수로 전달하도록 설정합니다.

while True: # 사용자가 종료할 때까지 화면을 계속 갱신하며 보여주는 무한 루프입니다.
    cv.imshow('Image', clone) # 사각형 드래그 현황이 반영된 clone 이미지를 창에 지속적으로 출력합니다.
    key = cv.waitKey(1) & 0xFF # 1밀리초 동안 키보드 입력을 기다리고, 입력받은 키 값을 가져옵니다.

    if key == ord('r'): # 키보드 'r' 키를 눌렀을 경우 (Reset)
        clone = img.copy() # 화면에 그려진 사각형을 지우기 위해 이미지를 초기 원본으로 되돌립니다.
        if cv.getWindowProperty("ROI", cv.WND_PROP_VISIBLE) >= 1: # 'ROI' 창이 현재 열려있는지 확인합니다.
            cv.destroyWindow("ROI") # 열려있다면 해당 창을 닫아 화면을 깨끗하게 만듭니다.
        roi = None # 저장되어 있던 잘라낸 이미지 데이터를 삭제합니다.
        print("영역 선택이 초기화되었습니다. 다시 드래그해 주세요.") # 콘솔에 상태를 알립니다.

    elif key == ord('s'): # 키보드 's' 키를 눌렀을 경우 (Save)
        if roi is not None: # 잘라낸 영역(ROI) 데이터가 메모리에 존재할 때만 실행합니다.
            cv.imwrite('saved_roi.jpg', roi) # 잘린 이미지를 'saved_roi.jpg'라는 이름의 파일로 현재 폴더에 저장합니다.
            print("선택한 영역이 'saved_roi.jpg'로 성공적으로 저장되었습니다.") # 저장 성공 메시지를 출력합니다.
        else: # 저장할 데이터가 없을 경우
            print("저장할 영역이 아직 선택되지 않았습니다.") # 경고 메시지를 출력합니다.

    elif key == ord('q'): # 키보드 'q' 키를 눌렀을 경우 (Quit)
        break # 무한 루프를 탈출하여 프로그램 종료 단계로 넘어갑니다.

cv.destroyAllWindows() # 프로그램 종료 전, 열려있던 모든 OpenCV 창을 닫고 메모리를 해제합니다.```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="2149" height="1240" alt="image" src="https://github.com/user-attachments/assets/e5c225fe-c24c-4e26-a9de-5897d9d1de5d" />
</div>
