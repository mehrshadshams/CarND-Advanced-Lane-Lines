import numpy as np
import cv2
import logging
from ring_buffer import RingBuffer


class Curvature(object):
    def __init__(self, pixel, world):
        self._pixel = pixel
        self._world = world

    @property
    def pixel(self):
        return self._pixel

    @property
    def world(self):
        return self._world


class Lane(object):
    def __init__(self, base, fit_params, fitx, curvature: Curvature):
        self._base = base
        self._fit_params = fit_params
        self._fitx = fitx
        self._curvature = curvature

    @property
    def base(self):
        return self._base

    @property
    def fit_params(self):
        return self._fit_params

    @property
    def fit_x(self):
        return self._fitx

    @property
    def curvature(self) -> Curvature:
        return self._curvature


class LaneFitter(object):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self, p_matrix, verbose=False):
        self._left: Lane = None
        self._right: Lane = None
        self._vebose = verbose
        self._M, self._M_inv = p_matrix
        self._buffer = RingBuffer(5)

    def _compute_average_base(self):
        items = self._buffer.get()
        if len(items) > 0:
            left_base = list(map(lambda x: x[0].base, items))
            right_base = list(map(lambda x: x[1].base, items))

            left_base = int(np.average(left_base))
            right_base = int(np.average(right_base))

            return left_base, right_base

        return None, None

    def _compute_average_fit(self, new_fit):
        items = self._buffer.get()

        left_fit = list(map(lambda x: x[0].fit_params, items))
        right_fit = list(map(lambda x: x[1].fit_params, items))

        if new_fit is not None and new_fit[0] is not None and new_fit[1] is not None:
            left_fit.append(new_fit[0])
            right_fit.append(new_fit[1])

        w = np.logspace(0, 1, len(left_fit))
        w /= sum(w)

        left_fit = np.average(np.array(left_fit), axis=0, weights=w)
        right_fit = np.average(np.array(right_fit), axis=0, weights=w)

        return left_fit, right_fit

    def _compute_curvature(self, ploty, fit, fitx) -> Curvature:
        # compute curvature

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image

        y_eval = np.max(ploty)

        curve_rad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

        # # Define conversions in x and y from pixels space to meters
        # ym_per_pix = 30 / imshape[0]  # meters per pixel in y dimension
        # xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        #
        # # Fit new polynomials to x,y in world space
        # fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
        #
        # # Calculate the new radii of curvature
        # curve_rad_real = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / \
        #                 np.absolute(2 * fit_cr[0])

        fit_cr = np.polyfit(ploty * LaneFitter.ym_per_pix, fitx * LaneFitter.xm_per_pix, 2)

        # Calculate the new radii of curvature
        curve_rad_world = ((1 + (2 * fit_cr[0] * y_eval * LaneFitter.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])

        return Curvature(curve_rad, curve_rad_world)

    def _compute_slope_mse(self, ploty, fit):
        left_fit, right_fit = fit
        left_slope = 2 * ploty * left_fit[0] + left_fit[1]
        right_slope = 2 * ploty * right_fit[0] + right_fit[1]
        mse = np.sum(np.power(left_slope - right_slope, 2)) / len(left_slope)
        return mse

    def fit_polynomial(self, y, x):
        if len(x) > 0 and len(y) > 0:
            return np.polyfit(y, x, 2)

        return None

    def draw_line(self, img, y, x):
        points = np.zeros((len(y), 2), dtype=np.int32)
        points[:, 1] = y.astype(np.int32)
        points[:, 0] = x.astype(np.int32)
        points = points.reshape((-1, 1, 2))

        cv2.polylines(img, points, True, (0, 255, 255), thickness=2)

    def fit(self, image):
        imshape = image.shape
        hist = np.sum(image[imshape[0] // 2:, :], axis=0)

        left_base, right_base = self._compute_average_base()

        midpoint = hist.shape[0] // 2

        if left_base is None:
            left_base = np.argmax(hist[:midpoint])

        if right_base is None:
            right_base = np.argmax(hist[midpoint:]) + midpoint

        out_img = np.dstack([image, image, image]) * 255
        n_windows = 9
        window_height = imshape[0] // n_windows
        margin = 100
        minpix = 50

        nonzero = image.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_lane_indices, right_lane_indices = [], []
        leftx_current = left_base
        rightx_current = right_base

        for window in range(n_windows):
            window_y_low = imshape[0] - (window + 1) * window_height
            window_y_high = imshape[0] - window * window_height
            window_x_left_low = leftx_current - margin
            window_x_left_high = leftx_current + margin
            window_x_right_low = rightx_current - margin
            window_x_right_high = rightx_current + margin

            cv2.rectangle(out_img, (window_x_left_low, window_y_low), (window_x_left_high, window_y_high), (0, 255, 0),
                          2)
            cv2.rectangle(out_img, (window_x_right_low, window_y_low), (window_x_right_high, window_y_high),
                          (0, 255, 0), 2)

            left_indices = ((nonzero_y >= window_y_low) & (nonzero_y < window_y_high) &
                            (nonzero_x >= window_x_left_low) & (nonzero_x <= window_x_left_high)).nonzero()[0]
            right_indices = ((nonzero_y >= window_y_low) & (nonzero_y < window_y_high) &
                             (nonzero_x >= window_x_right_low) & (nonzero_x <= window_x_right_high)).nonzero()[0]

            left_lane_indices.append(left_indices)
            right_lane_indices.append(right_indices)

            if len(left_indices) > minpix:
                leftx_current = np.int(np.mean(nonzero_x[left_indices]))
            if len(right_indices) > minpix:
                rightx_current = np.int(np.mean(nonzero_x[right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        leftx = nonzero_x[left_lane_indices]
        lefty = nonzero_y[left_lane_indices]
        rightx = nonzero_x[right_lane_indices]
        righty = nonzero_y[right_lane_indices]

        left_fit = self.fit_polynomial(lefty, leftx)
        right_fit = self.fit_polynomial(righty, rightx)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        # do sanity checking, are two lines relatively parallel?
        if left_fit is not None and right_fit is not None:
            slope_diff = self._compute_slope_mse(ploty, (left_fit, right_fit))
            if slope_diff > 0.1:
                print('ignoring fit')
                # ignore this fit
                left_fit, right_fit = None, None

        left_fit, right_fit = self._compute_average_fit((left_fit, right_fit))

        # Generate x and y values for plotting
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzero_y[left_lane_indices], nonzero_x[left_lane_indices]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_indices], nonzero_x[right_lane_indices]] = [0, 0, 255]

        self.draw_line(out_img, ploty, left_fitx)
        self.draw_line(out_img, ploty, right_fitx)

        left_curvature = self._compute_curvature(ploty, left_fit, left_fitx)
        right_curvature = self._compute_curvature(ploty, right_fit, right_fitx)

        if self._left is not None and self._right is not None:
            self._buffer.append((self._left, self._right))

        self._left = Lane(left_base, left_fit, left_fitx, left_curvature)
        self._right = Lane(right_base, right_fit, right_fitx, right_curvature)

        logging.info('Curvature (pixel space): {}'.format((self._left.curvature.pixel, self._right.curvature.pixel)))
        logging.info('Curvature (world space): {}'.format((self._left.curvature.world, self._right.curvature.world)))

        return out_img

    def transform(self, image):
        # Create an image to draw the lines on
        imshape = image.shape
        color_warp = np.zeros(imshape, dtype=np.uint8)

        left: Lane = self._left
        right: Lane = self._right

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left.fit_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right.fit_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        new_warp = cv2.warpPerspective(color_warp, self._M_inv, (imshape[1], imshape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, new_warp, 0.3, 0)

        return result, (left.curvature, right.curvature)

    def fit_transform(self, binary_image, color_image):
        fit_image = self.fit(binary_image)
        final_image, curvatures = self.transform(color_image)
        return final_image, fit_image, curvatures
