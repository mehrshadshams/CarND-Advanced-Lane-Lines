import numpy as np
import cv2
import matplotlib.pyplot as plt


class ConvolutionLaneFinder(object):
    def __init__(self, binary_image, original_image, p_matrix, verbose=False, output=None):
        self._image = binary_image
        self._original_image = original_image
        self._verbose = verbose
        self._output = output
        self._M, self._M_inv = p_matrix

    @staticmethod
    def window_mask(width, height, img_ref, center, level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
        max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
        return output

    def find_lanes(self):
        warped = self._image

        # window settings
        window_width = 50
        window_height = 80  # Break image into 9 vertical layers since image height is 720
        margin = 100  # How much to slide left and right for searching

        window_centroids = []

        window = np.ones(window_width)

        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2

        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        window_centroids.append((l_center, r_center))

        for level in range(1, int(warped.shape[0] / window_height)):
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :],
                axis=0)
            conv_signal = np.convolve(window, image_layer)

            offset = window_width / 2
            l_min_index = int(max(l_center - offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))

            if np.max(conv_signal[l_min_index:l_max_index]) > 0:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center - offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))

            if np.max(conv_signal[r_min_index:r_max_index]) > 0:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            window_centroids.append((l_center, r_center))

        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        for level in range(0, len(window_centroids)):
            l_mask = ConvolutionLaneFinder.window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = ConvolutionLaneFinder.window_mask(window_width, window_height, warped, window_centroids[level][1], level)

            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((warped, warped, warped))  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        nonzero = l_points.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_fit = np.polyfit(nonzero_y, nonzero_x, 2)

        nonzero = r_points.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        right_fit = np.polyfit(nonzero_y, nonzero_x, 2)

        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(output)
        ax.plot(left_fitx, ploty, color='red')
        ax.plot(right_fitx, ploty, color='blue')

        fig.savefig('images/{}_lines.jpg'.format(self._output))

        if self._verbose:
            plt.show()

        # compute curvature

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        print('Curvature (pixel space): {}'.format((left_curverad, right_curverad)))

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / warped.shape[0]  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                        np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                         np.absolute(2 * right_fit_cr[0])

        print('Curvature (real world space): ({}m, {}m)'.format(left_curverad, right_curverad))

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self._image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self._M_inv, (self._original_image.shape[1], self._original_image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self._original_image, 1, newwarp, 0.3, 0)

        text = 'Radius of Curvature = {}m'.format(int((left_curverad + right_curverad)/2))
        result = cv2.putText(img=result, text=text, org=(0, 50),
                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))

        return result, left_curverad, right_curverad


