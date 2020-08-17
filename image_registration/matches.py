import cv2 as cv

float_img = cv.imread('img/float.jpg', cv.IMREAD_GRAYSCALE)
ref_img = cv.imread('img/ref.jpg', cv.IMREAD_GRAYSCALE)

akaze = cv.AKAZE_create()
float_kp, float_des = akaze.detectAndCompute(float_img, None)
ref_kp, ref_des = akaze.detectAndCompute(ref_img, None)

# 特徴のマッチング
bf = cv.BFMatcher()
matches = bf.knnMatch(float_des, ref_des, k=2)

# 正しいマッチングのみ保持
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

matches_img = cv.drawMatchesKnn(
    float_img,
    float_kp,
    ref_img,
    ref_kp,
    good_matches,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('dest/matches.jpg', matches_img)
