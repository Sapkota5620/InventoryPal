import cv2
import numpy as np
from matplotlib import pyplot as pltd 

# web camera 

image_orig = cv2.imread('screenshot.jpg', -1)
#cv2.imshow("Image Orig", image_orig)

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

hsv = cv2.cvtColor(resized_up, cv2.COLOR_BGR2HSV)

lower = (12, 25, 25)
upper = (150, 255, 255)

mask = cv2.inRange(hsv, lower, upper)
color = cv2.bitwise_and(resized_up, resized_up, mask=mask)

#edge detection
img_blur = cv2.GaussianBlur(color, (3,3), 0)

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

#Display Sobel Edge Dection Images
#cv2.imshow("Sobel x", sobelx)
#cv2.waitKey(0)
#cv2.imshow("Sobel x", sobely)
#cv2.waitKey(0)
#cv2.imshow("Sobel xy", sobelxy)
#cv2.waitKey(0)
#cv2.imshow("ReSized Image Orig", color)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
cv2.imshow("Canny Edge Detection", color)
cv2.waitKey(0)









