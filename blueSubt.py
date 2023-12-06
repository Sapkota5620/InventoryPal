import cv2
import numpy as np
import sys

def nothing(x):
    pass

def boxesCounter(display, valid_boxes):
    for b_rect in valid_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), -1)


def analyze_boxes(matblobs, display_frame, size_threshold=1200):
    blobs, _ = cv2.findContours(matblobs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    wid = display_frame.shape[0]
    hei = display_frame.shape[1]
    for i, blob in enumerate(blobs):
        b_rect = cv2.boundingRect(blob)

        (rx, ry, rw, rh) = b_rect

        # Additional criteria for boxes can be added here if needed

        # Check if the area of the bounding rectangle is above the size threshold
        if (rw * rh > size_threshold) and ( rw * rh < int(wid/3 * hei/3)):
            valid_boxes.append(b_rect)
    
    if valid_boxes:
        print("Number of Large Boxes: ", len(valid_boxes))

        for i, b_rect in enumerate(valid_boxes):
            # Draw filled rectangles using cv2.rectangle
            cv2.rectangle(display_frame, (b_rect[0], b_rect[1]), (b_rect[0] + b_rect[2], b_rect[1] + b_rect[3]), (0, 255, 255), thickness=2)
            size = b_rect[2] * b_rect[3]
            cv2.putText(display_frame, f"{i + 1} : {size}", (b_rect[0], b_rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display_frame, f"{b_rect[2]} * {b_rect[3]}", (b_rect[0], b_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        cv2.imwrite("display_frame_in.jpg", display_frame)

    return valid_boxes


#cv2.namedWindow("B")
#cv2.namedWindow("C")
cv2.namedWindow("D")
'''
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
'''

"""
#set deafult value for saturation
cv2.createTrackbar("SAdd", "B", 0 , 255, nothing)
cv2.createTrackbar("SSub", "C", 0 , 255, nothing)
cv2.createTrackbar("VAdd", "C", 0 , 255, nothing)
cv2.createTrackbar("VSub", "C", 0 , 255, nothing)
"""
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

#cv2.createTrackbar("Th1", "C", 0 , 255, nothing)
#cv2.createTrackbar("Th2", "C", 0 , 255, nothing)

# web camera 
while True:
    image_orig = cv2.imread('images/5_ld_24.jpg')
    d_w = int(image_orig.shape[1] * (12/100))
    d_h = int(image_orig.shape[0] * (12/100))
    dim = (d_w, d_h)
    img = cv2.resize(image_orig, dim , interpolation= cv2.INTER_AREA)
    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    '''
    HMin = cv2.getTrackbarPos("HMin", "B")
    SMin = cv2.getTrackbarPos("SMin", "B")
    VMin = cv2.getTrackbarPos("VMin", "B")
    HMax = cv2.getTrackbarPos("HMax", "B")
    SMax = cv2.getTrackbarPos("SMax", "B")
    VMax = cv2.getTrackbarPos("VMax", "B")
    '''
    '''
    SAdd = cv2.getTrackbarPos("SAdd", "B")
    SSub = cv2.getTrackbarPos("SSub", "C")
    VAdd = cv2.getTrackbarPos("VAdd", "C")
    VSub = cv2.getTrackbarPos("VSub", "C")

    Th1 = cv2.getTrackbarPos("Th1", "C")
    Th2 = cv2.getTrackbarPos("Th2", "C")
    '''
    KernelSize = cv2.getTrackbarPos("KernelSize", "D")
    ErodeIter = cv2.getTrackbarPos("ErodeIter", "D")
    DilateIter = cv2.getTrackbarPos("DilateIter", "D")
    Canny1 = cv2.getTrackbarPos("Canny1", "D")
    Canny2 = cv2.getTrackbarPos("Canny2", "D")
    '''
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    lower = np.array([HMin, SMin, VMin])
    upper = np.array([HMax, SMax, VMax])

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(hsv, hsv, mask=mask)
    img_hsv = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    '''

    canny = cv2.Canny(gray, Canny1, Canny2)
    thresh = cv2.adaptiveThreshold(canny, 255, 1, 1, 3, 2) 
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    #applt dilation and erosion to join gaps 
    kernel = np.ones((KernelSize, KernelSize), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations= DilateIter)
    erode = cv2.erode(thresh, kernel, iterations = ErodeIter)


    countours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR) 

    images = [gray, blur, erode, dilated]
    names = ["gray", "blur", "edged", "dilated"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.drawContours(img, countours, -1, (0, 255, 0), 2)

    print('countours in the image:', len(countours))
    #cv2.imshow('HSV', img_hsv)
    cv2.imshow('Erode', erode)
    cv2.imshow('Dilated', dilated)

    cv2.imshow('Canny', img_copy)
    cv2.imshow('imag', img)

 
    
    
    cv2.waitKey(300)
