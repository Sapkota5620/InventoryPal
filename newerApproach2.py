import cv2
import matplotlib.pyplot as plt
import numpy as np
import cig_database as Cig


cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
file = "images/5_ld_24.jpg"
image = cv2.imread(file)
image = cv2.resize(image, (920, 920), interpolation= cv2.INTER_LINEAR) 

oriImage = image.copy()

def template_match(roi, template):
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]
    
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
                cv2.imwrite("roi.jpg", roi)
                cv2.destroyAllWindows()

                main_process()


def nothing(x):
   pass

# Morphological function sets
def morph_operation(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph

def analyze_boxes(matblobs, display_frame, size_threshold=5000):
    blobs, _ = cv2.findContours(matblobs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []
    wid = display_frame.shape[0]
    hei = display_frame.shape[1]
    for i, blob in enumerate(blobs):
        b_rect = cv2.boundingRect(blob)

        (rx, ry, rw, rh) = b_rect

        # Additional criteria for boxes can be added here if needed

        # Check if the area of the bounding rectangle is above the size threshold
        if (rw * rh > size_threshold) and ( rw * rh < int(wid/4 * hei/4)):
            if(rh > 65) and (rh < 85) and (rw > 100) * (rw < 140):
                valid_boxes.append(b_rect)
    
    if valid_boxes:
        print("Number of Large Boxes: ", len(valid_boxes))

        for i, b_rect in enumerate(valid_boxes):
            """ x, y, w, h = b_rect 
            roi = display_frame[y:y+h, x:x + w]
            cv2.imshow(f"Box {i + 1}", roi)

            """
            # Draw filled rectangles using cv2.rectangle
            cv2.rectangle(display_frame, (b_rect[0], b_rect[1]), (b_rect[0] + b_rect[2], b_rect[1] + b_rect[3]), (0, 255, 255), thickness=2)
            size = b_rect[2] * b_rect[3]
            cv2.putText(display_frame, f"{i + 1} : {size}", (b_rect[0], b_rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display_frame, f"{b_rect[2]} * {b_rect[3]}", (b_rect[0], b_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        cv2.imshow("display_frame_in", display_frame)

    return valid_boxes


def lbp_like_method(matinput, width_radius, height_radius, stren, off):
    height, width = np.shape(matinput)
    matdst = np.zeros_like(matinput)

    for y in range(height):
        y_min = max(0, y - height_radius)
        y_max = min(height, y + height_radius + 1)

        for x in range(width):
            x_min = max(0, x - width_radius)
            x_max = min(width, x + width_radius + 1)

            roi = matinput[y_min:y_max, x_min:x_max]
            center = matinput[y, x]

            valid_cells = roi[roi > center + off]
            total = len(valid_cells)

            if total > stren * (2 * height_radius + 2 * width_radius):
                matdst[y, x] = 255

    return matdst

def binary_inverse(img, minT=120, maxT=255):
    ret, thresh_gray = cv2.threshold(img, minT, maxT, cv2.THRESH_BINARY)
    # Inverse polarity
    thresh_gray = 255 - thresh_gray
    return thresh_gray

def boxesCounter(display, valid_boxes):
    for b_rect in valid_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), -1)


def main_process():
    """
    cv2.namedWindow("B")

    cv2.createTrackbar("w", "B", 1 , 200, nothing)
    cv2.createTrackbar("h", "B", 1 , 300, nothing)
    
    cv2.createTrackbar("peri", "B", 0 , 100, nothing)
    cv2.createTrackbar("off", "B", 0 , 20, nothing)
    cv2.createTrackbar("threshold", "B", 200 , 1200, nothing)
    cv2.createTrackbar("mul", "B", 1 , 5, nothing)
    cv2.setTrackbarPos("mul", "B", 1)
    """
    image_orig = cv2.imread('roi.jpg')
    #img = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR) 
    gray = cv2.cvtColor(image_orig,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),-1)
    
    """
    w = cv2.getTrackbarPos("w", "B") 
    h = cv2.getTrackbarPos("h", "B") 
    # Blured to remove noise 
    peri = cv2.getTrackbarPos("peri", "B")
    off = cv2.getTrackbarPos("off", "B")
    mul = cv2.getTrackbarPos("mul", "B")
    threshold = cv2.getTrackbarPos("threshold", "B")
    """

    # Parameter tuning
   # w = 1
    #h = 1
    peri = 0.82
    off = 15
    mul = 1
    threshold = 8000
    '''
    # optimized canny threshold 
    v = np.median(gray)
    sigma = 0.22
    #---- Apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))`

    '''
    canny = cv2.Canny(blurred,40,250)
    cv2.imshow("matlbp", canny)
    cv2.waitKey(0)


    thresh = cv2.adaptiveThreshold(canny, 255, 1, 1, 3, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    #thresh = cv2.dilate(thresh,None,iterations =1)
    #thresh = cv2.erode(thresh,None,iterations =1)

    matmorph = morph_operation(thresh)
    cv2.imshow("matlbp", matmorph)
    cv2.waitKey(0)


    """matlbp = lbp_like_method(gray, w*mul, h*mul, (peri/100), off * mul)
    cv2.imshow("matlbp", matlbp)
    cv2.waitKey(0)

    #canny = cv2.Canny(matlbp, 19, 215)
    matmorph = morph_operation(matlbp)

    canny_lbp = lbp_like_method(gray, w*mul, h*mul, (peri/100), off * mul)
    #bin = binary_inverse(matlbp, 120, 255)
    #bin2 = binary_inverse(matmorph, 120, 255)
    """
    
    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    valid_boxes = analyze_boxes(canny, display_color, size_threshold=threshold*mul) 
    boxesCounter(display_color, valid_boxes)

    """ 
    for b_rect in valid_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(display_color, (x, y), (x + w, y + h), (0, 255, 255), -1)
    

    titles = ['matlbp', 'canny', 'matmorph', 'canny_lbp','display_color']
    imagess = [matlbp, canny, matmorph, canny_lbp, display_color]

    for i in range(5):
        plt.subplot(2 , 3, i+1), plt.imshow(imagess[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()
    """
    cv2.imshow("matlbp", canny)
    cv2.waitKey(300)


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