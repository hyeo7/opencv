import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. 이미지 불러오기
# 요구사항: cv.imread()를 사용하여 두 개의 이미지를 불러옴
img1 = cv.imread('img1.jpg')
img2 = cv.imread('img2.jpg')

if img1 is None or img2 is None:
    print("이미지를 찾을 수 없습니다. 파일 경로 및 이름을 확인하십시오.")
else:
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 2. SIFT 특징점 검출
    # 요구사항: Cv.SIFT_create()를 사용하여 특징점을 검출
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. BFMatcher 특징점 매칭
    # 요구사항: cv.BFMatcher()와 knnMatch()를 사용하여 특징점을 매칭
    bf = cv.BFMatcher(cv.NORM_L2)
    knn_match = bf.knnMatch(des1, des2, k=2)

    # 4. 좋은 매칭점 선별 (Ratio Test)
    # 요구사항 힌트: 거리 비율이 임계값(예: 0.7) 미만인 매칭점만 선별
    good_match = []
    for m, n in knn_match:
        if m.distance < 0.7 * n.distance:
            good_match.append(m)

    # 5. 호모그래피 좌표 배열 추출
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

    # ====================================================================
    # 6. [핵심 교정] 투시 변환 방향 역전 (points2 -> points1)
    # ====================================================================
    # 오른쪽 이미지(img2)를 왼쪽 이미지(img1) 좌표계에 맞추기 위해 파라미터 순서를 변경합니다.
    # 요구사항 힌트: cv.RANSAC을 사용하면 이상점(Outlier) 영향을 줄일 수 있음
    H, mask = cv.findHomography(points2, points1, cv.RANSAC, 5.0)

    # 7. 이미지 투시 변환 및 정합
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 요구사항 힌트: 출력 크기를 두 이미지를 합친 파노라마 크기 (w1+w2, max(h1,h2))로 설정
    panorama_w = w1 + w2
    panorama_h = max(h1, h2)

    # 요구사항: cv.warpPerspective()를 사용하여 한 이미지를 변환하여 다른 이미지와 정렬
    # img2를 H 행렬에 따라 투시 변환합니다 (양수 X축 방향으로 이동됨).
    warped_img2 = cv.warpPerspective(img2, H, (panorama_w, panorama_h))
    
    # 변환된 캔버스의 (0,0) 원점 위치에 왼쪽 이미지인 img1을 덮어씌웁니다.
    panorama_result = warped_img2.copy()
    panorama_result[0:h1, 0:w1] = img1

    # 매칭 결과 시각화 준비
    matches_mask = mask.ravel().tolist()
    # 주의: findHomography에서 points2 -> points1 순서로 계산했으므로, 
    # drawMatches에서도 img2와 img1의 순서를 맞춰 그려주어야 매칭선이 꼬이지 않습니다.
    img_match = cv.drawMatches(img2, kp2, img1, kp1, good_match, None, 
                               matchesMask=matches_mask, 
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 8. Matplotlib 시각화
    # 요구사항: 변환된 이미지(Warped Image)와 특징점 매칭 결과(Matching Result)를 나란히 출력
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img_match, cv.COLOR_BGR2RGB))
    plt.title('Matching Result (RANSAC Inliers)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(panorama_result, cv.COLOR_BGR2RGB))
    plt.title('Warped Image (Panorama Alignment)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()