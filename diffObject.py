import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def get_edge(images):
    img_blur = cv.GaussianBlur(images, (5,5), 2)
    lap = cv.Laplacian(images, cv.CV_64F)
    cv.imshow("lap", lap)
    lap = np.uint8(np.absolute(lap))
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=255)
    return edges

img_rgb = cv.imread('roi.jpg')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('images/blu.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
template = cv.resize(template, (100, 80), interpolation= cv.INTER_LINEAR)

res = cv.matchTemplate(img_gray,template,cv.TM_SQDIFF_NORMED)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
threshold = 0.05
if max_val >= threshold:
    print('found, ')
    ned_h = h
    ned_w = w
    top_l = max_loc
    bot_r = (top_l[0] + ned_w, top_l[1] + ned_h)
    cv.rectangle(img_gray, top_l, bot_r, (0,255,255), 2, lineType=cv.LINE_4)

    cv.imshow("result", img_gray)
    cv.waitKey()
else:
    print("not found")