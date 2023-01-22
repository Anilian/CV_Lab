import cv2
import os
from glob import glob
import numpy as np
from random import randint
from skimage import util

def Resize_img(path2,down_width,down_height):
	for i, img in enumerate(glob("mixed/*.jpg"), 1):
	#for i, img in enumerate(glob("path/*"), 1):
		image = cv2.imread(img)
		down_points = (down_width, down_height)
		resize_down = cv2.resize(image, down_points, interpolation=cv2.INTER_LINEAR)
		img_resize = cv2.imwrite(os.path.join(path2, 'img_1280_{}.jpg'.format(i)), resize_down)
	return img_resize

# def fish_eye_effect(path2):
# 	for i, img in enumerate(glob("Plastic_resize/*.jpg"), 1):
#
# 		# link: http://zabaykin.ru/?p=262
#
# 		src = cv2.imread(img)
#
# 		h, w = src.shape[0:2]
# 		# получаем высоту и ширину изображения для
#
# 		# заполняем матрицу преобразования. сначала все нулями
# 		intrinsics = np.zeros((3, 3), np.float64)
#
# 		# матрица intrinsics
# 		intrinsics[0, 0] = 3500
# 		intrinsics[1, 1] = 3500
# 		intrinsics[2, 2] = 1.0
# 		intrinsics[0, 2] = w / 2.
# 		intrinsics[1, 2] = h / 2.
#
# 		newCamMtx = np.zeros((3, 3), np.float64)
# 		newCamMtx[0, 0] = 3500  # ширина
# 		newCamMtx[1, 1] = 3500
# 		newCamMtx[2, 2] = 1
# 		newCamMtx[0, 2] = w / 2.  # ширина
# 		newCamMtx[1, 2] = h / 2.
# 		# искажающие коэффициенты(их менять для изменения формы)
# 		dist_coeffs = np.zeros((1, 4), np.float64)
# 		dist_coeffs[0, 0] = 5.0  # закругление краев
# 		dist_coeffs[0, 1] = 0.0  # дальше: лево, право, вверх, вниз смещение
# 		dist_coeffs[0, 2] = 0.0
# 		dist_coeffs[0, 3] = -0.0
#
# 		map1, map2 = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, newCamMtx, src.shape[:2], cv2.CV_16SC2)
# 		res = cv2.remap(src, map1, map2, cv2.INTER_LINEAR)
#
# 		img_eff = cv2.imwrite(os.path.join(path2, 'image_fish eye_{}.jpg'.format(i)), res)
# 	return img_eff
def blur_motion_noise():
	for i, img in enumerate(glob("Plastic_change_resize/*.jpg"), 1):
		image = cv2.imread(img)
		###############  BLUR   #############
		blur = cv2.blur(image, (5, 5))
		img_blur = cv2.imwrite(os.path.join('Plastic_change_resize/blur', 'img_blur_1280_{}.jpg'.format(i)), blur)
		###############  MOTION BLUR   #############
		kernel_size = 10
		# Create the vertical kernel.
		kernel_v = np.zeros((kernel_size, kernel_size))
		# Create a copy of the same for creating the horizontal kernel.
		kernel_h = np.copy(kernel_v)
		# Fill the middle row with ones.
		kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
		kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
		# Normalize.
		kernel_v /= kernel_size
		kernel_h /= kernel_size
		# Apply the vertical kernel.
		vertical_mb = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_v)
		# Apply the horizontal kernel.
		#horizonal_mb = cv2.filter2D(img, -1, kernel_h)

		img_motion_blur = cv2.imwrite(os.path.join('Plastic_change_resize/motion blur', 'img_Mblur_1280_{}.jpg'.format(i)), vertical_mb)
		###############  NOISE   #############
		noise_img = util.random_noise(image, mode='s&p', amount=0.03)
		# The above function returns a floating-point image
		# on the range [0, 1], thus we changed it to 'uint8'
		# and from [0,255]
		noise_img = np.array(255 * noise_img, dtype='uint8')

		# Display the noise image
		cv2.imshow('blur', noise_img)
		img_motion_blur = cv2.imwrite(os.path.join('Plastic_change_resize/noise', 'Noise_1280_{}.jpg'.format(i)),noise_img)

	return img_blur, img_motion_blur, noise_img
# Resize_img('mixed_resize',1280,1280)
# fish_eye_effect("mixed_resize/fish eye effect")
blur_motion_noise()