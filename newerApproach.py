import cv2
import matplotlib.pyplot as plt
import numpy as np

# Morphological function sets
def morph_operation(matinput):
  kernel =  cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=2)
  morph = cv2.erode(matinput,kernel,iterations=1)
  morph = cv2.dilate(morph,kernel,iterations=1)

  return morph


"""# Analyze blobs
def analyze_blob(matblobs,display_frame):

  blobs,_ = cv2.findContours(matblobs,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
  valid_blobs = []

  for i,blob in enumerate(blobs):
    rot_rect = cv2.minAreaRect(blob)
    b_rect = cv2.boundingRect(blob)


    (cx,cy),(sw,sh),angle = rot_rect
    rx,ry,rw,rh = b_rect

    box = cv2.boxPoints(rot_rect)
    box = np.int0(box)

    # Draw the segmented Box region
    frame = cv2.drawContours(display_frame,[box],0,(0,0,255),1)

    on_count = cv2.contourArea(blob)
    total_count = sw*sh
    if total_count <= 0:
      continue

    if sh > sw :
      temp = sw
      sw = sh
      sh = temp

    # minimum area
    if sw * sh < 20:
      continue

    # maximum area
    if sw * sh > 100:
      continue  

    # ratio of box
    rect_ratio = sw / sh
    if rect_ratio <= 1 or rect_ratio >= 3.5:
      continue

    # ratio of fill  
    fill_ratio = on_count / total_count
    if fill_ratio < 0.4 :
      continue

    # remove blob that is too bright
    if display_frame[int(cy),int(cx),0] > 75:
      continue


    valid_blobs.append(blob)

  if valid_blobs:
    print("Number of Blobs : " ,len(valid_blobs))
  cv2.imshow("display_frame_in",display_frame)

  return valid_blobs
"""
def analyze_boxes(matblobs, display_frame, size_threshold=100):
    blobs, _ = cv2.findContours(matblobs, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_boxes = []

    for i, blob in enumerate(blobs):
        b_rect = cv2.boundingRect(blob)

        (rx, ry, rw, rh) = b_rect

        # Additional criteria for boxes can be added here if needed

        # Check if the area of the bounding rectangle is above the size threshold
        if rw * rh > size_threshold:
            valid_boxes.append(b_rect)

    if valid_boxes:
        print("Number of Large Boxes: ", len(valid_boxes))

        for b_rect in valid_boxes:
            # Draw filled rectangles using cv2.rectangle
            cv2.rectangle(display_frame, (b_rect[0], b_rect[1]), (b_rect[0] + b_rect[2], b_rect[1] + b_rect[3]), (0, 255, 255), thickness=cv2.FILLED)

        cv2.imshow("display_frame_in", display_frame)

    return valid_boxes



"""
def lbp_like_method(matinput,radius,stren,off):

  height, width = np.shape(matinput)

  roi_radius = radius
  peri = roi_radius * 8
  matdst = np.zeros_like(matinput)
  for y in range(height):
    y_ = y - roi_radius
    _y = y + roi_radius
    if y_ < 0 or _y >= height:
      continue


    for x in range(width):
      x_ = x - roi_radius
      _x = x + roi_radius
      if x_ < 0 or _x >= width:
        continue

      r1 = matinput[y_:_y,x_]
      r2 = matinput[y_:_y,_x]
      r3 = matinput[y_,x_:_x]
      r4 = matinput[_y,x_:_x]

      center = matinput[y,x]
      valid_cell_1 = len(r1[r1 > center + off])
      valid_cell_2 = len(r2[r2 > center + off])
      valid_cell_3 = len(r3[r3 > center + off])
      valid_cell_4 = len(r4[r4 > center + off])

      total = valid_cell_1 + valid_cell_2 + valid_cell_3 + valid_cell_4

      if total > stren * peri:
        matdst[y,x] = 255

  return matdst
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



def main_process():

  image_orig = cv2.imread('images/screenshot.jpg')
  img = cv2.resize(image_orig, (920, 920), interpolation= cv2.INTER_LINEAR) 
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Blured to remove noise 
  blurred = cv2.GaussianBlur(gray,(3,3),-1)


  """
  Implements a Local Binary Pattern (LBP)-like method. This method seems 
  to compute a binary pattern for each pixel in the input matrix (matinput)
  based on its local neighborhood.

  # Parameter tuning
  winsize = 5
  peri = 0.5
  off = 4

  matlbp = lbp_like_method(gray,winsize,peri,off)
  cv2.imshow("matlbp",matlbp)
  cv2.waitKey(1)
  """
  # Parameter tuning
  w = 7
  h = 4
  peri = 0.8
  off = 7

  matlbp = lbp_like_method(gray, w, h, peri,off)
  cv2.imshow("matlbp",matlbp)
  cv2.waitKey(1)

  matmorph = morph_operation(matlbp)
  cv2.imshow("matmorph",matmorph)
  cv2.waitKey(1)


  display_color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
  """valid_blobs = analyze_blob(matmorph,display_color)"""
  valid_boxes = analyze_boxes(matmorph, display_color, size_threshold=200) 


  """for b in range(len(valid_boxes)):
    cv2.drawContours(display_color,valid_boxes,b,(0,255,255),-1)
  """
  for b_rect in valid_boxes:
    x, y, w, h = b_rect
    cv2.rectangle(display_color, (x, y), (x + w, y + h), (0, 255, 255), -1)


  cv2.imshow("display_color",display_color)
  cv2.waitKey(0)
  

if __name__ == '__main__':
  main_process()