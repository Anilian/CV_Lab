import cv2
import numpy as np
import os

image = cv2.imread('C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\Mblue_sphere.jpg')
img_orig = cv2.imread('C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\image_24.jpg')

# Reading the image and converting into B/W
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

# Applying the function
fast = cv2.FastFeatureDetector_create(threshold=20)
fast.setNonmaxSuppression(True)

# Drawing the keypoints
kp = fast.detect(gray_image, None)
kp_orig = fast.detect(gray_image_orig, None)
print("Total Keypoints with nonmaxSuppression: ", len(kp))
print("Total Keypoints with nonmaxSuppression original image: ", len(kp_orig))

kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
kp_image_orig = cv2.drawKeypoints(img_orig, kp_orig, None, color=(0, 255, 0))

print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())

unified_image = np.hstack((kp_image_orig, kp_image))
cv2.namedWindow('FAST', cv2.WINDOW_NORMAL)
cv2.resizeWindow('FAST', 1000, 500)
cv2.imshow('FAST', unified_image)
path = 'C:\Personality\Stady\Lab_CV\Dataset\Results\FAST'
cv2.imwrite(os.path.join(path , 'FAST_Mblur_sphere_Threshold.jpg'),unified_image)
#+ repr(fast.getThreshold)+'Keypoints:'+ repr(len(kp)), img)
cv2.waitKey()