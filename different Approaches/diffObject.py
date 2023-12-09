# import required libraries
import cv2
import numpy as np

# load the input images
image1 = cv2.imread('images/src/Montego_Kings_red.jpg')
image2 = cv2.imread('images/src/247_Kings_green.jpg')
print(image1.shape[::-1])
print(image2.shape[::-1])
img1 = cv2.resize(image1, (458, 236), interpolation= cv2.INTER_LINEAR) 
img2 = cv2.resize(image2, (458, 236), interpolation= cv2.INTER_LINEAR) 
print(img1.shape[::-1])
print(img2.shape[::-1])
# convert the images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# define the function to compute MSE between two images
def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse, diff

error, diff = mse(img1, img2)
print("Image matching Error between the two images:",error)

cv2.imshow("difference", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()