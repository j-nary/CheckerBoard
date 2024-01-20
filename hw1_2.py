import cv2
import sys
import numpy as np

# 이미지 불러오기
# filename = 'board1.jpg'
if len(sys.argv) > 1:
    filename = sys.argv[1]
src = cv2.imread(filename)

if src is None:
    print('Image load failed!')
    exit()

# 마우스 이벤트 정의
src2 = src.copy()
def on_mouse(event, x, y, flags, param):
    global cnt, src_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if cnt < 4:
            src_pts[cnt, :] = np.array([x, y]).astype(np.float32)
            cnt += 1
            cv2.circle(src, (x, y), 5, (0, 0, 255) , -1)
            cv2.imshow('HW2_20212908', src)
        if cnt == 4:
            w = 500
            h = 500
            dst_pts = np.array([[0, 0],
                               [w - 1, 0],
                               [w - 1, h - 1],
                               [0, h - 1]]).astype(np.float32)
            pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            dst = cv2.warpPerspective(src2, pers_mat, (w, h))
            
            cv2.imshow('dst', dst)

cnt = 0
src_pts = np.zeros([4, 2], dtype=np.float32)

# 이벤트 등록하기
cv2.namedWindow('HW2_20212908')
cv2.setMouseCallback('HW2_20212908', on_mouse)

# 출력하기
cv2.imshow('HW2_20212908', src)
cv2.waitKey(0)
cv2.destroyAllWindows()