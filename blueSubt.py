import cv2
import numpy as np
import sys

# web camera 

image_orig = cv2.imread('screenshot.jpg', 0)
blu = cv2.imread(cv2.samples.findFile('blu.jpg'))

img = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

ret, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# find connected components
connectivity = 4
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity, cv2.CV_32S) 
sizes = stats[1:, -1]; nb_components = nb_components - 1
min_size = 250 #threshhold value for objects in scene
img2 = np.zeros((img.shape), np.uint8)
for i in range(0, nb_components+1):
    # use if sizes[i] >= min_size: to identify your objects
    color = np.random.randint(255,size=3)
    # draw the bounding rectangele around each object
    cv2.rectangle(img2, (stats[i][0],stats[i][1]),(stats[i][0]+stats[i][2],stats[i][1]+stats[i][3]), (0,255,0), 2)
    img2[output == i + 1] = color








