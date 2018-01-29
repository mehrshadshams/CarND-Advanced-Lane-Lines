import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    sobel = np.absolute(sobel)
    sobel = (255 * sobel / np.max(sobel)).astype(np.uint8)
    binary_output = np.zeros_like(sobel)
    binary_output[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = (255. * magnitude / np.max(magnitude)).astype(np.uint8)
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    direction = np.arctan2(sobely, sobelx)
    binary_output = np.zeros_like(gray)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    plt.imshow(binary_output, cmap='gray')
    return binary_output


def threshold_image(test_image, ksize=3, gradx_thresh=(20, 100), grady_thresh=(20, 100), magnitude_thresh=(30, 150),
                    dir_thresh=(0.9, 1.1)):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(test_image, orient='x', sobel_kernel=ksize, thresh=gradx_thresh)
    grady = abs_sobel_thresh(test_image, orient='y', sobel_kernel=ksize, thresh=grady_thresh)
    mag_binary = mag_thresh(test_image, sobel_kernel=ksize, thresh=magnitude_thresh)
    dir_binary = dir_threshold(test_image, sobel_kernel=ksize, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_color(image, thresh=(170, 255)):
    image_hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = image_hsl[:, :, 2]
    s_min = thresh[0]
    s_max = thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_min) & (s_channel <= s_max)] = 1
    return s_binary


def create_binary_image(image, ksize=3):
    gradient_threshold = threshold_image(image, ksize)
    color_threshold = threshold_color(image)

    binary = np.zeros_like(gradient_threshold)
    binary[(color_threshold == 1) | (gradient_threshold == 1)] = 1
    return binary


def show_image_gray(image):
    plt.imshow(image, cmap='gray')


def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


if __name__ == "__main__":
    image = mpimg.imread('test_images/test5.jpg')
    # image = create_binary_image(image, ksize=5)
    image = threshold_image(image)
    show_image_gray(image)
    plt.waitforbuttonpress()
