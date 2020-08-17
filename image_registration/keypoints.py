import cv2 as cv

# キーポイントなどを見やすくするためにグレースケールで画像読み込み
img = cv.imread('img/float.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# キーポイントの検出と特徴の記述
akaze = cv.AKAZE_create()
kp, descriptor = akaze.detectAndCompute(gray, None)

keypoints_img = cv.drawKeypoints(gray, kp, img)
cv.imwrite('dest/keypoints.jpg', keypoints_img)
