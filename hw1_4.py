import cv2
import numpy as np
import sys

def points(pts):    # 네 점 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]     # 좌상단
    rect[2] = pts[np.argmax(s)]     # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    return rect

def extract_checkerboard(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # 패턴의 경계 뚜렷하게
    # kernel = np.ones((5, 5), np.uint8)
    # binary = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    # cv.imshow('haaaaaaaaaaaaaaaaaaaaaaaaaa', binary)
    # cv.waitKey(0)

    # 이진화
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_not(binary)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 가장 큰 윤곽선 검출
    # largest_contour = max(contours, key=cv2.contourArea)
    # approx = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)

    # 큰 윤곽선부터 탐색
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in sorted_contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # cv.drawContours(image, [approx], -1, (0, 0, 255), 2)
        # cv.imshow('제발제발제발', image)
        # cv.waitKey(0)

        if len(approx) == 4:
            # 좌상단 -> 우상단 -> 우하단 -> 좌하단 순서로
            corners = points(approx.reshape(4, 2))

            # 최대 너비
            width = max([np.linalg.norm(corners[0] - corners[1]), np.linalg.norm(corners[2] - corners[3])])
            # 최대 높이
            height = max([np.linalg.norm(corners[0] - corners[3]), np.linalg.norm(corners[1] - corners[2])])
            
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            transform_matrix = cv2.getPerspectiveTransform(corners, dst)
            checkerboard = cv2.warpPerspective(image, transform_matrix, (int(width), int(height)))
            
            return checkerboard
    return None

def count_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=30, minRadius=20, maxRadius=35)

    light_count = 0
    dark_count = 0

    avgColor = np.mean(gray)

    ##타원 검출하기
    # for cnt in contours:
    #     # 외곽선의 면적을 계산
    #     area = cv2.contourArea(cnt)
    #     # 외곽선으로부터 최소 크기의 바운딩 서클(반지름)을 계산
    #     _, radius = cv2.minEnclosingCircle(cnt)

    #     # 최소 및 최대 반지름 조건을 확인
    #     if area >= np.pi * (minRadius ** 2) and area <= np.pi * (maxRadius ** 2) and radius >= minRadius and radius <= maxRadius:
    #         if len(cnt) >= 5:
    #             ellipse = cv2.fitEllipse(cnt)
    #             cv2.ellipse(resized_image, ellipse, (0, 255, 0), 2)

    if circles is not None:
        # 정수형 변환
        circles = np.round(circles[0, :]).astype("int")
        # print(f"총 원의 개수 : {len(circles)}")

        # for i in range(circles.shape[1]):
        #     cx, cy, radius = circles[0][i]
        #     cv2.circle(dst, (cx, cy), radius, (0,0,255), 2, cv2.LINE_AA)

        for (x, y, r) in circles:
            center_brightness = gray[y, x]
            if center_brightness > avgColor: # 밝은 색
                light_count = light_count + 1
                cv2.circle(image, (x, y), r, (255, 255, 255), 4)
            else: # 어두운 색
                dark_count = dark_count + 1
                cv2.circle(image, (x, y), r, (0, 0, 0), 4)

    return light_count, dark_count


if len(sys.argv) > 1:
    filename = sys.argv[1]
image = cv2.imread(filename)

if image is None:
    print('Image load failed!')
    exit()

# cv2.imshow('HW4_20212908', image)

checkerboard = extract_checkerboard(image)
if checkerboard is not None:
    checkerboard = cv2.resize(checkerboard, (600, 600))
    # cv2.imshow('dst', checkerboard)

light_count, dark_count= count_circle(checkerboard)
print(f"w:{light_count} b:{dark_count}")

# cv2.waitKey(0)
# cv2.destroyAllWindows()