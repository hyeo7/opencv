<div align="center">

# 컴퓨터 비전 6주차 실습
이번 실습은 Python과 OpenCV, MediaPipe를 활용하여 <br>
SORT 알고리즘을 이용한 다중 객체 추적기 구현과, <br>
MediaPipe를 활용한 얼굴 랜드마크 추출 및 시각화에 대한 실습입니다. <br>

</div>
<br><br>

# 1. SORT 알고리즘을 활용한 다중 객체 추적기 구현 (6.1.py)
[실습 목표]<br>
손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기를 구현합니다.
<br><br>
[과제 설명]<br>
YOLOv3와 같은 사전 훈련된 객체 검출 모델을 사용하여 각 프레임에서 객체를 검출하고, <br> 
검출된 객체의 경계 상자를 입력으로 받아 SORT 추적기를 초기화합니다. <br>
이후 각 프레임마다 검출된 객체와 기존 추적 객체를 연관시켜 추적을 유지하며, <br>
추적된 각 객체에 고유 ID를 부여하여 해당 ID와 경계 상자를 실시간으로 출력합니다. <br>

<details>
<summary><b>전체 코드 및 주석 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2
import numpy as np
import os
import sys

try:
    from sort import Sort
except ImportError:
    print("에러: sort.py 파일을 찾을 수 없습니다. GitHub에서 다운로드하여 동일 폴더에 배치하세요.")
    sys.exit()

# 1. 경로 및 설정값 정의
CFG_FILE = 'yolov3.cfg'
WEIGHTS_FILE = 'yolov3.weights' # 용량이 커서 별도 준비된 파일
VIDEO_FILE = 'slow_traffic_small.mp4'
CONF_THRESHOLD = 0.5  # 실습 자료 요구사항: Confidence > 0.5
NMS_THRESHOLD = 0.4   # 실습 자료 요구사항: NMS Threshold 0.4

# 2. 모델 및 추적기 초기화
def initialize_tracker():
    # [요구사항 1] 객체 검출기 구현 (YOLOv3 모델 로드)
    net = cv2.dnn.readNetFromDarknet(CFG_FILE, WEIGHTS_FILE)
    
    # GPU 가속 설정 (환경에 따라 선택적 사용)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # [요구사항 2] SORT 추적기 초기화
    # max_age: 객체를 놓쳤을 때 유지할 프레임 수 (ID 유지를 위해 10으로 상향 조정)
    # min_hits: 추적 시작을 위한 최소 검출 횟수
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)
    
    return net, output_layers, tracker

def process_video():
    net, output_layers, mot_tracker = initialize_tracker()
    cap = cv2.VideoCapture(VIDEO_FILE)

    if not cap.isOpened():
        print(f"에러: 비디오 파일({VIDEO_FILE})을 열 수 없습니다.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        height, width, _ = frame.shape

        # [요구사항 1-1] 각 프레임에서 객체 검출 (Blob 전처리)
        # 실습 자료 규격: 1/255 스케일링, 416x416 크기, RB 채널 스왑
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 검출 데이터 파싱
        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # 'car'(인덱스 2) 클래스에 대해 설정된 Confidence 이상만 수집
                if class_id == 2 and confidence > CONF_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # SORT 입력 규격인 [x1, y1, x2, y2]로 변환
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)
                    boxes.append([x1, y1, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # [요구사항 1-2] Non-Maximum Suppression 적용
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        detections_for_sort = []
        for i in indices:
            # 인덱스 접근 방식 호환성 처리 (Flatten)
            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
            x, y, w, h = boxes[idx]
            # SORT는 [x1, y1, x2, y2, score] 형태의 입력을 기대함
            detections_for_sort.append([x, y, x + w, y + h, confidences[idx]])

        # [요구사항 3] 객체 추적 (SORT 데이터 연관)
        # 칼만 필터 예측과 헝가리안 알고리즘 매칭 수행
        if len(detections_for_sort) > 0:
            trackers = mot_tracker.update(np.array(detections_for_sort))
        else:
            trackers = mot_tracker.update(np.empty((0, 5)))

        # [요구사항 4] 결과 시각화 (고유 ID 부여 및 박스 표시)
        for d in trackers:
            x1, y1, x2, y2, obj_id = map(int, d)
            
            # 이미지 65a577.png 처럼 시각화 구성
            color = (0, 255, 0) # 초록색 박스
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 결과 출력
        cv2.imshow('Dynamic Vision: Multi-Object Tracking (SORT)', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
</div>
<br><br>

2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화 (6.2.py)
[실습 목표] <br>
Mediapipe의 FaceMesh 모듈을 사용하여 얼굴의 468개 랜드마크를 추출하고, 이를 실시간 영상에 시각화하는 프로그램을 구현합니다.
<br><br>
[과제 설명] <br>
Mediapipe의 solutions.face_mesh를 사용하여 얼굴 랜드마크 검출기를 초기화합니다. <br>
OpenCV를 사용하여 웹캠으로부터 실시간 영상을 캡처한 뒤, <br>
검출된 랜드마크 좌표를 변환하여 실시간 영상에 얼굴 그물망, 경계, 눈동자 등으로 세분화하여 시각화합니다. <br>
키보드 'q' 키를 누르면 프로그램이 안전하게 종료되도록 제어합니다. <br>

<details>
<summary><b>전체 코드 및 주석 보기 (클릭하여 펼치기)</b></summary>
<div markdown="1">

```python
import cv2
import sys

try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except ImportError as e:
    print(f"라이브러리 로드 오류: {e}")
    sys.exit()

# [요구사항 1] FaceMesh 초기화
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# [요구사항 2] 웹캠 실시간 영상 캡처
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # BGR -> RGB 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # [힌트 3] 정규화 좌표 변환을 위한 이미지 크기 획득
            ih, iw, _ = frame.shape

            # [요구사항 3] 랜드마크 시각화
            for landmark in face_landmarks.landmark:
                # [힌트 3] 정규화 좌표 -> 픽셀 좌표 변환
                x_px = int(landmark.x * iw)
                y_px = int(landmark.y * ih)
                # [힌트 2] circle 함수 사용
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

    cv2.imshow('Final Fix - FaceMesh', frame)

    # [요구사항 4] ESC 종료 (27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()```
```


</div>
</details>

[실행 결과 화면]
<div align="center">
<img width="933" height="691" alt="image" src="https://github.com/user-attachments/assets/d412e9e9-298b-42a9-a6af-2329f46cf7db" />
</div>
<br><br>
