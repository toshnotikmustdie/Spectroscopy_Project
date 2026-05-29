import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('tea_spectrum.jpg')
if img is None:
    sys.exit("Could not read the image.")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
intensities = np.array(gray)
column = intensities.mean(axis=0)

plt.plot(column)
plt.xlabel('Position (px)')
plt.ylabel('Intensity')
plt.show()