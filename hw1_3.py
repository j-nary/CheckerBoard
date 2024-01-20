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


if len(sys.argv) > 1:
    filename = sys.argv[1]
image = cv2.imread(filename)

if image is None:
    print('Image load failed!')
    exit()

cv2.imshow('HW3_20212908', image)

checkerboard = extract_checkerboard(image)
if checkerboard is not None:
    checkerboard = cv2.resize(checkerboard, (600, 600))
    cv2.imshow('dst', checkerboard)

cv2.waitKey(0)
cv2.destroyAllWindows()