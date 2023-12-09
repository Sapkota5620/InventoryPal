import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

image_orig = cv2.imread(cv2.samples.findFile('images/screenshot2.jpg'))
#cv2.imshow("Image Orig", image_orig)
image_orig2 = cv2.imread(cv2.samples.findFile('images/box.png'))
#resizing image
'''
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image_orig, down_points, interpolation= cv2.INTER_LINEAR)
''' 

resized_up = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR)

#cv2.imshow("ReSized Image Orig", resized_down)
#cv2.waitKey()

hsv1 = cv2.cvtColor(resized_up, cv2.COLOR_BGR2GRAY)
hsv2 = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
'''
# using corner matching
corners = cv2.goodFeaturesToTrack(hsv2, 100, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(image_orig, (x,y), 5, (0,255,0), -1)

for i in range(len(corners)):
    for j in range(i + 1, len(corners)):
        corners1 = tuple(corners[i][0])
        corners2 = tuple(corners[j][0])
        color = tuple(map(lambda x: int(x), np.random.randint(0, 255, size=3)))
        cv2.line(image_orig2, corners1, corners2, color, 1)
'''

#feature detection SIFT
orb = cv2.SIFT_create()
kp1, des1 = orb.detectAndCompute(hsv1, None)
kp2, des2 = orb.detectAndCompute(hsv2, None)

imgKp1 = cv2.drawKeypoints(resized_up, kp1, None)
imgKp2 = cv2.drawKeypoints(image_orig2, kp1, None)

#brute force matcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(resized_up, kp1,image_orig2,kp2,good,None, flags=2)


#cv2.imshow('1', imgKp1)
#cv2.imshow('2', imgKp2)
cv2.imshow('3', img3)
cv2.waitKey(0)
"""
lower = (12, 25, 25)
upper = (150, 255, 255)

ret, thresh = cv2.threshold(hsv2, 45, 255, 0)
mask = cv2.inRange(hsv1, lower, upper)
color = cv2.bitwise_and(resized_up, resized_up, mask=mask)

#edge detection
img_blur = cv2.GaussianBlur(color, (3,3), 0)

# Laplacian Method
lap = cv2.Laplacian(resized_up, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

# Sobel XY Method
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
sobelC = cv2.bitwise_or(sobelx, sobely)

# Canny edge dector - multistage algo to detect wide range of edge in image
edges = cv2.Canny(image=lap, threshold1=100, threshold2=200)

#countor edge detection
count, heih = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
count_image = cv2.drawContours(resized_up, count, -1, (0, 255, 0), 3)
"""
"""# Using matplotlib to display picture
titles = ['img', 'hsv', 'mask', 'lap', 'sobelxy', 'sobelC', 'edges', 'count']
imagess = [image_orig, hsv, mask, lap, sobelxy, sobelC, edges, count_image]

for i in range(7):
    plt.subplot(2 , 4, i+1), plt.imshow(imagess[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()


"""





