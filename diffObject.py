import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def get_edge(images):
    img_blur = cv.GaussianBlur(images, (5,5), 2)
    lap = cv.Laplacian(images, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=255)
    return edges

img_rgb = cv.imread('images/screenshot.jpg')
assert img_rgb is not None, "file could not be read, check with os.path.exists()"
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template = cv.imread('images/blu.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
w, h = template.shape[::-1]
img = cv.resize(img_gray, (960, 960), interpolation= cv.INTER_LINEAR)

img_rgb_edge = get_edge(img)
img_template_edge = get_edge(template)
#cv.imshow("1", img_rgb_edge)
cv.imshow("2", img_template_edge)

res = cv.matchTemplate(img_rgb_edge,img_template_edge,cv.TM_SQDIFF_NORMED)
cv.imshow("3" , res)

cv.waitKey(0)

threshold = 0.05
loc = np.where( res <= threshold)
for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb_edge, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
cv.imwrite('res.png',img_rgb_edge)