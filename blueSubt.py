import cv2
import numpy as np
import sys

def nothing(x):
    pass

cv2.namedWindow("B")
cv2.namedWindow("C")
cv2.namedWindow("D")

#scale of HSV H:0-179, S: 0-255, V:0-255
cv2.createTrackbar("HMin", "B", 0 , 179, nothing)
cv2.createTrackbar("SMin", "B", 0 , 255, nothing)
cv2.createTrackbar("VMin", "B", 0 , 255, nothing)
cv2.createTrackbar("HMax", "B", 0 , 179, nothing)
cv2.createTrackbar("SMax", "B", 0 , 255, nothing)
cv2.createTrackbar("VMax", "B", 0 , 255, nothing)

#set deafult value for Max HSV Ttacker
cv2.setTrackbarPos("HMax", "B", 179)
cv2.setTrackbarPos("SMax", "B", 255)
cv2.setTrackbarPos("VMax", "B", 255)
#set deafult value for saturation
cv2.createTrackbar("SAdd", "B", 0 , 255, nothing)
cv2.createTrackbar("SSub", "C", 0 , 255, nothing)
cv2.createTrackbar("VAdd", "C", 0 , 255, nothing)
cv2.createTrackbar("VSub", "C", 0 , 255, nothing)

#edge creation
cv2.createTrackbar("KernelSize", "D", 1 , 30, nothing)
cv2.createTrackbar("ErodeIter", "D", 1 , 5, nothing)
cv2.createTrackbar("DilateIter", "D", 1 , 5, nothing)
cv2.createTrackbar("Canny1", "D", 0 , 200, nothing)
cv2.createTrackbar("Canny2", "D", 0 , 500, nothing)
#Set Defaults
cv2.setTrackbarPos("KernelSize", "D", 1)
cv2.setTrackbarPos("Canny1", "D", 100)
cv2.setTrackbarPos("Canny2", "D", 200)

cv2.createTrackbar("Th1", "C", 0 , 255, nothing)
cv2.createTrackbar("Th2", "C", 0 , 255, nothing)

# web camera 
while True:
    image_orig = cv2.imread('images/24_edgefield.jpg')
    d_w = int(image_orig.shape[1] * (12/100))
    d_h = int(image_orig.shape[0] * (12/100))
    dim = (d_w, d_h)
    img = cv2.resize(image_orig, dim , interpolation= cv2.INTER_AREA)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    HMin = cv2.getTrackbarPos("HMin", "B")
    SMin = cv2.getTrackbarPos("SMin", "B")
    VMin = cv2.getTrackbarPos("VMin", "B")
    HMax = cv2.getTrackbarPos("HMax", "B")
    SMax = cv2.getTrackbarPos("SMax", "B")
    VMax = cv2.getTrackbarPos("VMax", "B")

    SAdd = cv2.getTrackbarPos("SAdd", "B")
    SSub = cv2.getTrackbarPos("SSub", "C")
    VAdd = cv2.getTrackbarPos("VAdd", "C")
    VSub = cv2.getTrackbarPos("VSub", "C")

    Th1 = cv2.getTrackbarPos("Th1", "C")
    Th2 = cv2.getTrackbarPos("Th2", "C")

    KernelSize = cv2.getTrackbarPos("KernelSize", "D")
    ErodeIter = cv2.getTrackbarPos("ErodeIter", "D")
    DilateIter = cv2.getTrackbarPos("DilateIter", "D")
    Canny1 = cv2.getTrackbarPos("Canny1", "D")
    Canny2 = cv2.getTrackbarPos("Canny2", "D")

    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    lower = np.array([HMin, SMin, VMin])
    upper = np.array([HMax, SMax, VMax])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(hsv, hsv, mask=mask)
    img_hsv = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    kernel = np.ones((KernelSize, KernelSize), np.uint8)
    erode = cv2.erode(img_hsv, kernel, iterations = ErodeIter)
    dilated = cv2.dilate(erode, kernel, iterations= DilateIter)
    canny = cv2.Canny(dilated, Canny1, Canny2)
    
    img_copy = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    
    '''
    #countours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #Modify image shape for stacking
    np.stack((mask,) * 3, axis=-1)
    np.stack((gray,) * 3, axis=-1)
    np.stack((blur,) * 3, axis=-1)
    np.stack((erode,) * 3, axis=-1)
    np.stack((dilated,) * 3, axis=-1)

    images = [mask, gray, blur, erode, dilated]
    names = ["mask","gray", "blur", "edged", "dilated"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    img_stack = np.hstack(images)

   # cv2.drawContours(img, countours, -1, (0, 255, 0), 2)

    #print('countours in the image:', len(countours))
    '''
    #cv2.imshow('Input', img_stack)
    cv2.imshow('HSV', img_hsv)
    cv2.imshow('Erode', erode)
    cv2.imshow('Dilated', dilated)
    cv2.imshow('Canny', img_copy)
    
    
    cv2.waitKey(300)
