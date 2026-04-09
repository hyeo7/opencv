import cv2
import sys

# [수정] mp.solutions를 거치지 않고 내부 경로를 직접 강제로 가져옵니다.
# 파이썬 3.13에서 발생하는 속성 참조 오류(AttributeError)를 원천 차단합니다.
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    print("성공: MediaPipe 모듈을 심층 경로에서 직접 로드했습니다.")
except ImportError:
    print("에러: 라이브러리 내부 파일에 접근할 수 없습니다.")
    sys.exit()

# [요구사항 1] FaceMesh 모듈 초기화
# mp.solutions.face_mesh 대신 mp_face_mesh를 직접 사용합니다.
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,        # 실시간 처리를 위해 False
    max_num_faces=1,                # 468개 랜드마크 추출 대상 얼굴 수
    refine_landmarks=True,          # 눈/입 주변 정밀화 활성화
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# [요구사항 2] 웹캠으로부터 실시간 영상 캡처
cap = cv2.VideoCapture(0)

print("실습 02 실행 중... 종료하려면 ESC 키를 누르세요.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 전처리: Mediapipe는 RGB 형식을 사용함
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 추출 수행
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # [힌트 3] 이미지 크기 획득하여 정규화 좌표 변환 준비
            ih, iw, _ = frame.shape

            # [요구사항 3] 검출된 랜드마크를 실시간 영상에 점으로 표시
            for landmark in face_landmarks.landmark:
                # [힌트 3] 정규화 좌표(0~1)를 픽셀 좌표로 변환
                x_px = int(landmark.x * iw)
                y_px = int(landmark.y * ih)

                # [힌트 2] OpenCV의 circle 함수를 사용하여 시각화
                # 반지름 1, 초록색(0, 255, 0), 두께 -1(채우기)
                cv2.circle(frame, (x_px, y_px), 1, (0, 255, 0), -1)

    # 결과 영상 출력
    cv2.imshow('Mediapipe FaceMesh (Fix for Python 3.13)', frame)

    # [요구사항 4] ESC 키를 누르면 프로그램 종료 (ASCII 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
face_mesh.close()