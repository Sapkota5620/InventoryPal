import numpy as np
import cv2

# Read input image as Grayscale
orig = cv2.imread('images/screenshot.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(orig, (920, 920), interpolation= cv2.INTER_LINEAR)

# Convert img to uint8 binary image with values 0 and 255
# All white pixels above 250 goes to 255, and other pixels goes to 0
ret, thresh_gray = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)

# Inverse polarity
thresh_gray = 255 - thresh_gray


# Find contours in thresh_gray.
contours = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]  # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

corners = []

# Iterate contours, find bounding rectangles, and add corners to a list
for c in contours:
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(c)

    # Append corner to list of corners - format is corners[i] holds a tuple: ((x0, y0), (x1, y1))
    corners.append(((x, y), (x+w, y+h)))

# Convert grayscale to BGR (just for testing - for drawing rectangles in green color).
out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw green rectangle (for testing)
for c in corners:
    cv2.rectangle(out, c[0], c[1], (0, 255, 0), thickness = 2)

cv2.imwrite('out.png', out)  #Save out to file (for testing).


# Show result (for testing).
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()