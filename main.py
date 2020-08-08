import cv2
import numpy as np
import matplotlib.pyplot as plt

impath = "./dataset/mod/mod_1.jpg"

img = cv2.imread(impath)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## HSV Masking
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([15, 70, 245], dtype=np.uint8)
upper = np.array([25, 80, 255], dtype=np.uint8)

mask = cv2.inRange(hsv, lower, upper)


## Morphological transformations on masked image
kernel = np.ones((3, 3), dtype=np.uint8)
tf = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

contours, hierarchy = cv2.findContours(tf, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


contour = max(contours, key=cv2.contourArea)
boundingRect = cv2.boundingRect(contour)

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