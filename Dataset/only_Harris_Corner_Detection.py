import cv2
import numpy as np



img = cv2.imread('C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\image_24.jpg')
cv2.namedWindow('Harris Corner Detection Test', cv2.WINDOW_NORMAL)

def f(x=None):
    return

cv2.createTrackbar('Harris Window Size', 'Harris Corner Detection Test', 9, 25, f)
cv2.createTrackbar('Harris Parameter', 'Harris Corner Detection Test', 1, 100, f)
cv2.createTrackbar('Sobel Aperture', 'Harris Corner Detection Test', 3, 14, f)
cv2.createTrackbar('Detection Threshold', 'Harris Corner Detection Test', 3, 100, f)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

img_bak = img

while True:
    img = img_bak.copy()

    window_size = cv2.getTrackbarPos('Harris Window Size', 'Harris Corner Detection Test')
    harris_parameter = cv2.getTrackbarPos('Harris Parameter', 'Harris Corner Detection Test')
    sobel_aperture = cv2.getTrackbarPos('Sobel Aperture', 'Harris Corner Detection Test')
    threshold = cv2.getTrackbarPos('Detection Threshold', 'Harris Corner Detection Test')
    print(window_size,harris_parameter,sobel_aperture,threshold)
    sobel_aperture = sobel_aperture * 2 + 1

    if window_size <= 0:
        window_size = 1

    dst = cv2.cornerHarris(gray, window_size, sobel_aperture, harris_parameter/100)

    # Result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > threshold/100 * dst.max()] = [0, 0, 255]

    dst_show = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    dst_show = (255*dst_show).astype(np.uint8)
    unified_image =  np.hstack((img, dst_show))
    cv2.imshow('Harris Corner Detection Test', unified_image)
    # cv2.imwrite('results_HCD/imageqqqqq.jpg', unified_image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
