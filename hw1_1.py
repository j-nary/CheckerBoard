import math
import cv2
import sys
import numpy as np


def count_grid(image):
    # 명암 높이기
    alpha = 1.5
    beta = 0
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거를 위한 가우시안 블러 적용
    image = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('dst', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 이진화
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 윤곽선 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 행과 열 개수 계산
        rows = len(np.array(approx[:, :, 1]))
        cols = len(np.array(approx[:, :, 0]))
        
        # 윤곽선 정확성 판단 기준값
        acThreshold = 0.9
        # 행과 열이 체스보드 패턴에 부합하면 출력
        # if 8 <= rows <= 12 and 8 <= cols <= 12:
        # print(cols, rows)
        if 6 <= rows <= 12 and 6 <= cols <= 12 and cv2.arcLength(approx, True) > acThreshold * cv2.arcLength(contour, True):
            # cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)
            # cv2.imshow('Contours', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            if rows >= 9:
                rows = 10;
                cols = 10;
            else:
                rows = 8;
                cols = 8;  
            return rows, cols

# 이미지 불러오기
# filename = 'board1.jpg'
if len(sys.argv) > 1:
    filename = sys.argv[1]
image = cv2.imread(filename)

if image is None:
    print('Image load failed!')
    exit()

# cv2.imshow('HW1_20212908', image)

rows, cols = count_grid(image)
print(f"체스판 크기: {rows} x {cols}")

# cv2.waitKey(0)
# cv2.destroyAllWindows()