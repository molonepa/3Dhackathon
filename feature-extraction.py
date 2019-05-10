import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_LSH = 6

bim = cv2.imread('bim2.png')
bim_gray= cv2.cvtColor(bim,cv2.COLOR_BGR2GRAY)

image = cv2.imread('image2.jpg')
image=cv2.resize(image, (1280, 720))
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
bim_kp,bim_desc =orb.detectAndCompute(bim_gray,None)
image_kp,image_desc = orb.detectAndCompute(gray,None)


index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(bim_desc,image_desc,k=2)
good = []

for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>=MIN_MATCH_COUNT:
    src_pts = np.float32([ bim_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ image_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w,_ = bim.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    image = cv2.polylines(image,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(bim,bim_kp,image,image_kp,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()
cv2.waitKey(0)
