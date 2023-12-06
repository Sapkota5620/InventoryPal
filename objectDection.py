import cv2
import numpy as np


cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
file = "images/5_ld_24.jpg"
image = cv2.imread(file)
image = cv2.resize(image, (920, 920), interpolation= cv2.INTER_LINEAR) 

oriImage = image.copy()

def template_match(roi, template):
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, oriImage
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start, y_start), (x_end, y_end)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            if roi is not None and roi.size > 0:  # Check if roi is not empty
                cv2.imshow("Cropped", roi)
                cv2.imwrite("roi1.jpg", roi)
                cv2.destroyAllWindows()

                main_process()


def main_process():
    # Load the image
    file = "roi1.jpg"
    img = cv2.imread(file)
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    """ 
    v = np.median(gray)
    sigma = .55
    #---- Apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    """
    
    edged = cv2.Canny(gray, 19, 230)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 3, 2)
    cv2.imshow("win_name2",thresh)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations =1)
    thresh = cv2.erode(thresh,None,iterations =1)
   

    # Find the contours
    contours,hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >30000:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img,
                        (x,y),(x+w,y+h),
                        (0,255,0),
                        5)
        
    cv2.imshow("win_name",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        i = image.copy()
        if not cropping:
            cv2.imshow("image", image)
        elif cropping:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)
        key = cv2.waitKey(1)
        if key == 27:  # 27 is the ASCII code for the Escape key
            break
    cv2.destroyAllWindows()