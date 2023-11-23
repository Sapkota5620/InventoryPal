import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
# web camera 

image_orig = cv2.imread(cv2.samples.findFile('screenshot.jpg'))
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
hsv1 = cv2.cvtColor(resized_up, cv2.COLOR_BGR2GRAY)

lower = (12, 25, 25)
upper = (150, 255, 255)

ret, thresh = cv2.threshold(hsv1, 45, 255, 0)
mask = cv2.inRange(hsv, lower, upper)
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

# Using matplotlib to display picture
titles = ['img', 'img_blur', 'resized_up' ]
imagess = [image_orig, hsv,  resized_up]
'''
for i in range(3):
    plt.subplot(1 , 1, i+1), plt.imshow(imagess[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
'''
print(len(count))
plt.imshow(count_image, 'gray')
plt.show()









