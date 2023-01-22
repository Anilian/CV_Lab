import cv2
import os

#usefull link https://stackoverflow.com/questions/48063525/error-with-matches1to2-with-opencv-sift
#######Brute-Force Matching with SIFT Descriptors and Ratio Test######
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

def read_image(path1, path2):
    # reading the images from their using imread() function
    read_img1 = cv2.imread(path1)
    read_img2 = cv2.imread(path2)
    return (read_img1, read_img2)


# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1, pic2):
    gray_img1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return (gray_img1, gray_img2)


if __name__ == '__main__':
    first_image_path = 'C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\Mblue_sphere.jpg'
    second_image_path = 'C:\Personality\Stady\Lab_CV\Dataset\Plastic_resize\diff\image_24.jpg'
    # reading the image from there path by calling the function
    img1, img2 = read_image(first_image_path, second_image_path)

    # converting the read images into the gray scale images
    gray_pic1, gray_pic2 = convert_to_grayscale(img1, img2)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_pic1, None)
    kp2, des2 = sift.detectAndCompute(gray_pic2, None)
    print("Total Keypoints: ", len(kp1))
    print("Total Keypoints original image: ", len(kp2))

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    good_without_list = []
    for m,n in matches:
        if m.distance < 0.25*n.distance: #здесь можно изменять количество лучших точек
            good.append([m])
            good_without_list.append(m)
    print("Total Keypoints after sorting: ", len(good))

    knn_image = cv2.drawMatchesKnn(img2,kp1,img1,kp2,good,None, flags=2)


    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', 1000, 500)
    cv2.imshow('output', knn_image)
    path = 'C:\Personality\Stady\Lab_CV\Dataset\Results\FAST'
    cv2.imwrite(os.path.join(path, 'BBrute-Force Matching_SHIFT_sphere_Threshold.jpg'), knn_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
