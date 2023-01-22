import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
from pathlib import Path

#link: http://zabaykin.ru/?p=262

path = 'Plastic_resize/motion blur/img_Mblur_1280_17.jpg'
src = cv2.imread(path)

h, w = src.shape[0:2]
# получаем высоту и ширину изображения для


# заполняем матрицу преобразования. сначала все нулями
intrinsics = np.zeros((3, 3), np.float64)

# матрица intrinsics
intrinsics[0, 0] = 3500
intrinsics[1, 1] = 3500
intrinsics[2, 2] = 1.0
intrinsics[0, 2] = w / 2.
intrinsics[1, 2] = h / 2.


newCamMtx = np.zeros((3, 3), np.float64)
newCamMtx[0, 0] = 3500 #ширина
newCamMtx[1, 1] = 3500
newCamMtx[2, 2] = 1
newCamMtx[0, 2] = w / 2.#ширина
newCamMtx[1, 2] = h / 2.
#искажающие коэффициенты(их менять для изменения формы)
dist_coeffs = np.zeros((1, 4), np.float64)
dist_coeffs[0, 0] = 5.0 #закругление краев
dist_coeffs[0, 1] = 0.0 #дальше: лево, право, вверх, вниз смещение
dist_coeffs[0, 2] = 0.0
dist_coeffs[0, 3] = -0.0


map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, newCamMtx, src.shape[:2], cv2.CV_16SC2)
res = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
img_eff = cv2.imwrite('Plastic_resize/diff/QQQQ.jpg',res)

# ###############  MOTION BLUR   #############
# path = 'Mixed_resize/img_1280_1.jpg'
# image = cv2.imread(path)
# kernel_size = 10
# # Create the vertical kernel.
# kernel_v = np.zeros((kernel_size, kernel_size))
# # Create a copy of the same for creating the horizontal kernel.
# kernel_h = np.copy(kernel_v)
# # Fill the middle row with ones.
# kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
# kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
# # Normalize.
# kernel_v /= kernel_size
# kernel_h /= kernel_size
# # Apply the vertical kernel.
# vertical_mb = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_v)
# # Apply the horizontal kernel.
# #horizonal_mb = cv2.filter2D(img, -1, kernel_h)
# img_motion_blur = cv2.imwrite('mixed_resize/motion blur/img_Mblur_1280_2.jpg', vertical_mb)

