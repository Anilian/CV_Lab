# Importing the libraries
# только нахождения ключевых точек, без сметчинга
import cv2
import numpy as np
import os

# Reading the image and converting into B/W
image = cv2.imread('C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\Mblue_sphere.jpg')
img_orig = cv2.imread('C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\image_24.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

# Applying the function
orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_image, None)
kp_orig, des_orig = orb.detectAndCompute(gray_image_orig, None)

# Drawing the keypoints
kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)
kp_image_orig = cv2.drawKeypoints(img_orig, kp_orig, None, color=(0, 255, 0), flags=0)
print("Total Keypoints with nonmaxSuppression: ", len(kp))
print("Total Keypoints with nonmaxSuppression original image: ", len(kp_orig))

unified_image = np.hstack((kp_image_orig, kp_image))
cv2.namedWindow('ORB', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ORB', 1000, 500)
cv2.imshow('ORB', unified_image)
path = 'C:\Personality\Stady\Lab_CV\Dataset\Results\ORB'
# cv2.imwrite(os.path.join(path , 'ORB_Mblur_sphere_Threshold.jpg'),unified_image)

cv2.waitKey()