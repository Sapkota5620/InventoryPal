import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

''''
source: Temple Matching OpenCv
https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html

'''

img = cv.imread('images/src/247_100s_red.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img2 = img.copy()
template = cv.imread('images/box/cropped1.jpg', cv.IMREAD_GRAYSCALE)
assert template is not None, "file could not be read, check with os.path.exists()"
wi, hi = img.shape[::-1]
w, h = template.shape[::-1]
img  = cv.resize(img, (w , h), cv.INTER_AREA )
cv.imshow("1", img)
cv.imshow("2", template)
cv.waitKey(0)

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
for meth in methods:
    img = img2.copy()
    method = eval(meth)
    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        loc = np.where(res <= min_val + 0.05)  # Threshold for SQDIFF methods
    else:
        loc = np.where(res >= max_val - 0.05)  # Threshold for other methods
    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + w, pt[1] + h)
        cv.rectangle(img, pt, bottom_right, (0,255,0), 2)

    result_filename = f'result_{meth}.png'
    cv.imwrite(result_filename, img)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()