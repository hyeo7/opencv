import cv2
import numpy as np
import os
import sys

# [준비사항] filterpy, scikit-image 설치 및 sort.py 파일이 동일 경로에 있어야 함
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
    process_video()