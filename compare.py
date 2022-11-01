import sys

from skimage.metrics import structural_similarity
import cv2
import numpy as np


photo = cv2.imread('photo2.jpg')
passport = cv2.imread('passport.jpg')

photo = cv2.resize(photo, (passport.shape[1], passport.shape[0]))
photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
passport = cv2.cvtColor(passport, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = structural_similarity(photo, passport, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

