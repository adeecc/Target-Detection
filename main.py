import cv2
import numpy as np
import matplotlib.pyplot as plt

impath = "./dataset/sim/sim_1.jpg"

image = cv2.imread(impath)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.plot(th)

# th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find Canny edges
edged = cv2.Canny(th, 30, 200)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged,
                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# plt.imshow(edged)
# cv2.waitKey(0)

print(hierarchy)
print(f"Number of Contours found = {len(contours)}")

# Draw all contours
# -1 signifies drawing all contours

# get largest contour
# contour = contours[1]
contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
# cv2.drawContours(image, [contour], 0, (0, 255, 0), 3)

plt.imshow(image)
plt.show()

# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
