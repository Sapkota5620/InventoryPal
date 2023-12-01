#Github Approach
#https://stackoverflow.com/questions/58305151/how-to-use-opencv-to-count-the-stack-of-the-boxes

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import math

image_orig = cv2.imread('images/screenshot.jpg')
img = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR)
kernel1 = np.ones((5, 5), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)
#kernel3 = np.ones((5, 5), np.uint8)
#img = cv2.dilate(img, kernel1, iterations=1)
#img = cv2.erode(img, kernel2, iterations=1)

show = cv2.erode(img, kernel1, iterations=1)
show = cv2.dilate(img, kernel2, iterations=1)
opening = cv2.morphologyEx(show, cv2.MORPH_OPEN, kernel1)
canny = cv2.Canny(opening, 30, 120)


lines = cv2.HoughLinesP(canny, 1, np.pi / 200, 90, minLineLength=20, maxLineGap=10)
for line in range(0, len(lines)):
    x1, y1, x2, y2 = lines[line][0]
    # cv2.line(show, (x1, y1), (x2, y2), (255, 0, 0), 2)
# cv2.imshow('first', show)
result = []


# cannot delete directly from the array because inside the  for loop
# use dummy "result[]" to keep the lines that needed
# return the result back to the array after the for loop

for line in range(0, len(lines)):
    x1, y1, x2, y2 = lines[line][0]
    if x1 == x2:
        continue
    angle = math.atan(float((y2 - y1)) / float((x2 - x1)))
    angle = angle * 180 / math.pi
    # print(angle)
    if abs(angle) <= 5 and ((y1 or y2) < (show.shape[0] - 30)):
        result.append(lines[line][0])
lines = result


print(len(lines))
data = []
for line in range(0, len(result)):
    x1, y1, x2, y2 = lines[line]
    cv2.line(show, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #cv2.imshow('show2', show)
    data.append((y1 + y2) / 2)

cv2.imshow('show2', show)
cv2.waitKey(0)