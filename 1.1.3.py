import cv2 as cv

# 1. 축구 이미지 불러오기
img = cv.imread('soccer.jpg') 

# 이미지가 제대로 불러와졌는지 확인
if img is None:
    print("이미지를 불러올 수 없습니다. 파일명과 경로를 확인해 주세요.")
    exit()

# 초기 붓 크기는 5를 사용 
brush_size = 5
drawing_left = False
drawing_right = False

# 마우스 이벤트는 cv.setMouseCallback()을 통해 처리하며, cv.circle()을 이용해 현재 붓 크기로 원을 그림 [cite: 40]
def paint(event, x, y, flags, param):
    global brush_size, drawing_left, drawing_right, img

    # 좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기 
    if event == cv.EVENT_LBUTTONDOWN:
        drawing_left = True
        cv.circle(img, (x, y), brush_size, (255, 0, 0), -1) 

    elif event == cv.EVENT_RBUTTONDOWN:
        drawing_right = True
        cv.circle(img, (x, y), brush_size, (0, 0, 255), -1) 

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing_left:
            cv.circle(img, (x, y), brush_size, (255, 0, 0), -1)
        elif drawing_right:
            cv.circle(img, (x, y), brush_size, (0, 0, 255), -1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing_left = False
    elif event == cv.EVENT_RBUTTONUP:
        drawing_right = False

cv.namedWindow('Paint')
cv.setMouseCallback('Paint', paint)

# Key 입력은 루프 안에서 처리 [cite: 42]
while True:
    cv.imshow('Paint', img)
    
    # cv.waitKey(1)로 받은 값을 이용해 +, -, q를 구분 [cite: 41]
    key = cv.waitKey(1) & 0xFF

    # q 키를 누르면 영상 창이 종료 
    if key == ord('q'):
        break
        
    # + 입력 시 붓 크기 1 증가 
    elif key == ord('+') or key == ord('='):
        # 붓 크기는 최대 15로 제한 
        brush_size = min(15, brush_size + 1)
        print(f"현재 붓 크기: {brush_size}")
        
    # - 입력 시 붓 크기 1 감소 
    elif key == ord('-'):
        # 붓 크기는 최소 1로 제한 
        brush_size = max(1, brush_size - 1)
        print(f"현재 붓 크기: {brush_size}")

cv.destroyAllWindows()