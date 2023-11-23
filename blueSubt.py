import cv2
import numpy as np
import sys

def nothing(x):
    print(x)

cv2.namedWindow("B")
cv2.createTrackbar("Th1", "B", 0 , 255, nothing)
cv2.createTrackbar("Th2", "B", 0 , 255, nothing)

# web camera 
while True:
    image_orig = cv2.imread('screenshot.jpg')
    img = cv2.resize(image_orig, (480, 360), interpolation= cv2.INTER_LINEAR)
    img_copy = img.copy()

    th1 = cv2.getTrackbarPos("Th1", "B")
    th2 = cv2.getTrackbarPos("Th2", "B")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    edged = cv2.Canny(blur, th1, th2, 3)
    dilated = cv2.dilate(edged, (1,1), iterations= 2)

    countours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Modify image shape for stacking
    np.stack((gray,) * 3, axis=-1)
    np.stack((blur,) * 3, axis=-1)
    np.stack((edged,) * 3, axis=-1)
    np.stack((dilated,) * 3, axis=-1)

    images = [gray, blur, edged, dilated]
    names = ["gray", "blur", "edged", "dilated"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_stack = np.hstack(images)

    cv2.drawContours(img, countours, -1, (0, 255, 0), 2)

    print('countours in the image:', len(countours))

    cv2.imshow('Input', img_stack)
    cv2.imshow('Output',img)
    cv2.waitKey(300)
