import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(rows, cols, img_list):
    pass

impath = "./dataset/mod/mod_1.jpg"

img = cv2.imread(impath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# TODO: Try Blur
# TODO: Create Plots



## HSV Masking
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([15, 70, 245], dtype=np.uint8)
upper = np.array([25, 80, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower, upper)
masked = cv2.bitwise_and(img, img, mask=mask)



## Morphological transformations on masked image
kernel = np.ones((3, 3), dtype=np.uint8)

# tf = cv.erode(mask, kernel, iterations=1)
# tf = cv.dilate(mask, kernel, iterations=1)
tf = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)



## Thresholding
tf_thmean = cv2.adaptiveThreshold(tf, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
# tf_thgaus = cv2.adaptiveThreshold(tf, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)



## Canny Edge Detection
# TODO: Do canny edge detection

# contours, hierarchy = cv2.findContours(tf, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(tf_thmean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


contour = max(contours, key=cv2.contourArea)
boundingRect = cv2.boundingRect(contour)

# _ = cv.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.rectangle(rgb, boundingRect, color=(255, 0, 255), thickness=4)

cv2.circle(
    rgb, 
    center=(boundingRect[0] + boundingRect[2] // 2, 450), 
    radius=5, 
    color=(255, 0, 0), 
    thickness=5
)

plt.imshow(rgb)
plt.show()