import cv2
import matplotlib.pyplot as plt
import numpy as np
from cig_database import Cigarette # database for all brands and their pictures


cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0
file = "images/24_edgefield.jpg"
image = cv2.imread(file)
image = cv2.resize(image, (920, 920), interpolation= cv2.INTER_LINEAR) 

oriImage = image.copy()

global_boxes = []


"""
    Crops the image into order to create the 
    best size for image processing.
    Uses mouse clicks and draws a rectangle
    for easy visualization.

"""    

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
                cv2.imwrite("roi.jpg", roi)
                cv2.destroyAllWindows()

                main_process()


# Morphological function sets
def morph_operation(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph


"""
    Calculates for redundencies in the global list
    use IOU (Intersection over Union)
"""
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    intersection_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection_area = intersection_x * intersection_y
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / max(1e-5, union_area)
    return iou
"""
    Given an colored image, input matrix, grayscale_image, size_threshold, iou_threshold,
    1) findCountours in the image
    2) createa a bounding box
    3) check if the box is an appropriate size
    4) check if is already exist in the list of boxes
    5) crops the images to store it for later comparison
    6) numbers them in order
    7) returns list of boxes's coordinate
"""
def analyze_boxes( colored_image, matblobs, display_frame, size_threshold=5000, iou_threshold = 0.03):
    global global_boxes

    blobs, _ = cv2.findContours(matblobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                #Check for duplicates based on Iou
                is_duplicate = any(calculate_iou(b_rect, existing_box) > iou_threshold for existing_box in global_boxes)
                if not is_duplicate:
                    valid_boxes.append(b_rect)
                    global_boxes.append(b_rect)
    if valid_boxes:
        print("Number of Large Boxes: ", len(valid_boxes))

        for i, b_rect in enumerate(valid_boxes):
            
            x, y, w, h = b_rect 
            #roi = display_frame[y - 15: y + h + 15, x - 15:x + w + 15]
            roi = colored_image[y - 15: y + h + 15, x - 15:x + w + 15]
            cv2.imwrite(f"images/box/cropped{i + 1}.jpg", roi)
            # Draw filled rectangles using cv2.rectangle
            cv2.rectangle(display_frame, (b_rect[0], b_rect[1]), (b_rect[0] + b_rect[2], b_rect[1] + b_rect[3]), (0, 255, 255), thickness=2)
            size = b_rect[2] * b_rect[3]
            cv2.putText(display_frame, f"{i + 1} : {size}", (b_rect[0], b_rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(display_frame, f"{b_rect[2]} * {b_rect[3]}", (b_rect[0], b_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        #cv2.imshow("display_frame_in", display_frame)

    return valid_boxes


    """
    Implements a Local Binary Pattern (LBP)-like method. Compute a 
    binary pattern for each pixel in the input matrix (matinput) 
    based on its local neighborhood. 
    Parameter tuning:
    1. matinput- gray scale input matrix
    2. width_radius, height_radius - radius of local neighborhood around each pixel.
    3. stren - level of intensity
    4. off - offset value for reducing distant and ungrouped pixels (noise)
    """

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

"""
    With given image and list of tuples (x, y, w, h),
    it draws an rectangle in the given image
"""
def boxesCounter(display, valid_boxes):
    for b_rect in valid_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), -1)


def generate_text_to_image(boxes, most_similar_product, best_similarity, similarity_scores, hist_template_scores, gray_image):
    # Define font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    line_height = 20
    starting_height = 30

    # Write the text on the gray image
    cv2.putText(gray_image, f"The most boxes counted are {boxes}!",
                (10, starting_height), font, font_scale, 0, font_thickness * 2, cv2.LINE_AA)
    starting_height += line_height * 2            
    cv2.putText(gray_image, f"The most similar product is '{most_similar_product}' with a similarity index of {best_similarity}",
                (10, starting_height), font, font_scale, 0, font_thickness, cv2.LINE_AA)

    starting_height += line_height * 2  # Increase the starting height for the next block of text
    cv2.putText(gray_image, "Similarity scores:", (10, starting_height), font, font_scale, 0, font_thickness, cv2.LINE_AA)
    starting_height += line_height  # Increase the starting height for the next block of text

    for product_name, similarity_index in similarity_scores.items():
        cv2.putText(gray_image, f"{product_name}: {similarity_index}",
                    (10, starting_height), font, font_scale, 0, font_thickness, cv2.LINE_AA)
        starting_height += line_height

    starting_height += line_height  # Increase the starting height for the next block of text
    cv2.putText(gray_image, "Best template scores:", (10, starting_height), font, font_scale, 0, font_thickness, cv2.LINE_AA)
    starting_height += line_height  # Increase the starting height for the next block of text
    
    for product_name, best_template_score in hist_template_scores.items():
        cv2.putText(gray_image, f"{product_name}: {best_template_score}",
                    (10, starting_height), font, font_scale, 0, font_thickness, cv2.LINE_AA)
        starting_height += line_height
    
    starting_height += line_height * 2
    cv2.putText(gray_image, f"Press ESC to exit the windows!",
            (10, starting_height), font, font_scale, 0, font_thickness * 2, cv2.LINE_AA)

    return gray_image

def main_process():
    cig = Cigarette()
    image_orig = cv2.imread('roi.jpg', 1)

    gray = cv2.cvtColor(image_orig,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),-1)

    width = 5 
    height = 3
    strenght = 86
    offset = 8
    threshold = 8000

    matlbp = lbp_like_method(gray, width, height, (strenght / 100), offset)
    matmorph = morph_operation(matlbp)
    
    # optimized canny threshold 
    v = np.median(gray)
    sigma = 0.32
    #---- Apply automatic Canny edge detection using the computed median----
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    canny_o = cv2.Canny(gray,lower,upper)
    thresh_o = cv2.adaptiveThreshold(canny_o, 255, 1, 1, 3, 2)


    # Simple Methond  of Canny-> threshold -> Dilate -> erode
    canny = cv2.Canny(gray,60,180)
    thresh = cv2.adaptiveThreshold(canny, 255, 1, 1, 3, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations =1)
    thresh = cv2.erode(thresh,None,iterations =1)

   
    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    colored_image = image_orig
    valid_boxes = analyze_boxes(colored_image, thresh_o, display_color, size_threshold=threshold) 
    boxesCounter(display_color, valid_boxes)

    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    colored_image = image_orig
    valid_boxes = analyze_boxes(colored_image, thresh, display_color, size_threshold=threshold) 
    boxesCounter(display_color, valid_boxes)

    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    colored_image = image_orig
    valid_boxes = analyze_boxes(colored_image, matmorph, display_color, size_threshold=threshold) 
    boxesCounter(display_color, valid_boxes)
    
    print("Maximum boxes that were counted: ", len(global_boxes))
    colored_image = image_orig
    display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    for b_rect in global_boxes:
        x, y, w, h = b_rect
        cv2.rectangle(colored_image, (x, y), (x + w, y + h), (255, 255, 0), thickness=4)
    cv2.imshow("Final Result", colored_image)
    image_paths_to_compare = []
    for i in range(len(global_boxes)):
        image_paths_to_compare.append(f'images/box/cropped{i+1}.jpg')
    
    # Call the function
    most_similar_product, best_similarity, similarity_scores, hist_template_scores = cig.find_most_similar_productV3(image_paths_to_compare)
    blank_page = np.ones((400, 800), dtype=np.uint8) * 200
    generate_text_to_image(len(global_boxes), most_similar_product, best_similarity, similarity_scores, hist_template_scores, blank_page)
    
    cv2.imshow("Text to Screen", blank_page)
    cv2.waitKey(0)

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