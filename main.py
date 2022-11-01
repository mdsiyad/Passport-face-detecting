# import cv2
# import numpy as np
#
# original = cv2.imread("photo2.jpg")
# duplicate = cv2.imread("second.jpg")  # 1) Check if 2 images are equals
# if original.shape == duplicate.shape:
#     print("The images have same size and channels")
#     difference = cv2.subtract(original, duplicate)
#
#     b, g, r = cv2.split(difference)
#
#     if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
#         print("The images are completely Equal")
#
# else:
#     print("the images are not equal")
import cv2
import numpy as np

# read image 1
img1 = cv2.imread('person1.jpg')

# read image 2
img2 = cv2.imread('person2.JPG')

# do absdiff
diff = cv2.absdiff(img1, img2)

# get mean of absdiff
mean_diff = np.mean(diff)

# print result
print(mean_diff)
