import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity
import cv2
import numpy as np
from deepface import DeepFace
import pytesseract
# import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"



# Get user supplied values
imagePath = "images/passport.jpg"


# cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
# faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     #flags = cv2.CV_HAAR_SCALE_IMAGE
# )

# print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# crop_passport = image[y:y + h, x:x + w]

# img1 = cv2.imread('images/ismacil.jpeg')
# plt.imshow(img1[:, :, ::-1])
# plt.show()
# img1 = DeepFace.detectFace(imagePath)
# plt.imshow(img1)
# plt.imshow(img1[:, :, ::-1])
# plt.show()

# cv2.imshow("passport", crop_passport)
# cv2.waitKey(0)
# img2 = cv2.imread('images/cadaawe.jpg')
# plt.imshow(img2[:, :, ::-1])
# plt.show()

# metrics = ["cosine", "euclidean", "euclidean_l2"]

# models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]

# result = DeepFace.verify(img1, img2,enforce_detection=False)
# obj = DeepFace.analyze(img1, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)

# print('Is same face: ', result['verified'])
# print(obj)
# print(result)
