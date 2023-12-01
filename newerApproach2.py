import cv2
import matplotlib.pyplot as plt
import numpy as np


def nothing(x):
   pass

# Morphological function sets
def morph_operation(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph

def analyze_boxes(matblobs, display_frame, size_threshold=200):
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

def main_process():
  cv2.namedWindow("B")

  cv2.createTrackbar("w", "B", 1 , 15, nothing)
  cv2.createTrackbar("h", "B", 1 , 15, nothing)
  cv2.createTrackbar("peri", "B", 0 , 100, nothing)
  cv2.createTrackbar("off", "B", 0 , 20, nothing)
  cv2.createTrackbar("threshold", "B", 200 , 1200, nothing)
  cv2.createTrackbar("mul", "B", 1 , 5, nothing)
  cv2.setTrackbarPos("mul", "B", 1)  
  image_orig = cv2.imread('images/8_ld_edgefield.jpg')
  img = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR) 
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  while(True):
    # Blured to remove noise 
    blurred = cv2.GaussianBlur(gray,(3,3),-1)
    w = cv2.getTrackbarPos("w", "B") 
    h = cv2.getTrackbarPos("h", "B") 
    peri = cv2.getTrackbarPos("peri", "B")
    off = cv2.getTrackbarPos("off", "B")
    mul = cv2.getTrackbarPos("mul", "B")
    """
    # Parameter tuning
    w = 7
    h = 4
    peri = 0.8
    off = 7
    """
    matlbp = lbp_like_method(gray, w*mul, h*mul, (peri/100),off * mul)
    cv2.imshow("matlbp",matlbp)
    cv2.waitKey(1)
    
    bin = binary_inverse(matlbp, 120, 255)
    cv2.imshow("b_i_matlp", bin)
    cv2.waitKey(1)


    matmorph = morph_operation(matlbp)
    cv2.imshow("matmorph",matmorph)
    cv2.waitKey(1)

    bin2 = binary_inverse(matmorph, 120, 255)
    cv2.imshow("b_i_matmorph", bin2)
    cv2.waitKey(1)

    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    threshold = cv2.getTrackbarPos("threshold", "B")

    valid_boxes = analyze_boxes(matmorph, display_color, size_threshold=threshold*mul) 


    for b_rect in valid_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(display_color, (x, y), (x + w, y + h), (0, 255, 255), -1)


    cv2.imshow("display_color",display_color)
    cv2.waitKey(10)
  

if __name__ == '__main__':
  main_process()