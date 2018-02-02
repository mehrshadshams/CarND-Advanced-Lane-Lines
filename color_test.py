import cv2
import numpy as np
import matplotlib.pyplot as plt


def thresh_bin_im(image):
    """
    Return the colour thresholds binary for L, S and R channels in an image
    img: RGB image
    """

    def bin_it(image, threshold):
        output_bin = np.zeros_like(image)
        output_bin[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return output_bin

    # convert image to hls colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    # binary threshold values
    bin_thresh = [20, 255]

    # rgb thresholding for yellow
    lower = np.array([225, 180, 0], dtype="uint8")
    upper = np.array([255, 255, 170], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    rgb_y = bin_it(rgb_y, bin_thresh)

    # rgb thresholding for white (best)
    lower = np.array([100, 100, 200], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    rgb_w = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
    rgb_w = bin_it(rgb_w, bin_thresh)

    # hls thresholding for yellow
    lower = np.array([20, 120, 80], dtype="uint8")
    upper = np.array([45, 200, 255], dtype="uint8")
    mask = cv2.inRange(hls, lower, upper)
    hls_y = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    hls_y = bin_it(hls_y, bin_thresh)

    im_bin = np.zeros_like(hls_y)
    im_bin[(hls_y == 1) | (rgb_y == 1) | (rgb_w == 1)] = 1

    return im_bin


# if __name__ == "__main__":
#     image = cv2.imread('test_images/test5.jpg')
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     bin = thresh_bin_im(image)
#
#     plt.imshow(bin, cmap='gray')
#     plt.waitforbuttonpress()
