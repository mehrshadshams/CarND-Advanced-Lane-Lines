import argparse
import cv2
import numpy as np
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import os

from utils import threshold_image, threshold_color, undistort_image, show_image_gray
from find_lanes import ConvolutionLaneFinder


OUT_DIR = "output_images"
IMAGES_DIR = "images"
PERSPECTIVE_FILE = "perspective"


class Pipeline(object):
    def __init__(self, args):
        print("Loading '{}'".format(args.filename))
        image = cv2.imread(args.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self._verbose = args.verbose
        self._image = image
        self._file_name_no_ext = os.path.splitext(os.path.split(args.filename)[-1])[0]

        if self._verbose and not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)

        print('Loading threshold parameters...')
        with open('parameters.json', 'r') as f:
            self._thresh_params = json.load(f)['adjusted']

        print('Loading camera calibration parameters...')
        with open('camera.pkl', 'rb') as f:
            camera = pickle.load(f)

        self._mtx, self._dist = camera['mtx'], camera['dist']

        self._image_undist = undistort_image(image, self._mtx, self._dist)

    def apply_thresholing(self):
        parameters = self._thresh_params
        image = self._image_undist

        gradx_thresh = (parameters['sobel_x_thresh_min'], parameters['sobel_x_thresh_max'])
        grady_thresh = (parameters['sobel_y_thresh_min'], parameters['sobel_y_thresh_max'])
        magnitude_thresh = (parameters['mag_thresh_min'], parameters['mag_thresh_max'])
        dir_thresh = (parameters['dir_thresh_min'], parameters['dir_thresh_max'])

        gradient_threshold = threshold_image(image, ksize=parameters['sobel_kernel'],
                                             gradx_thresh=gradx_thresh, grady_thresh=grady_thresh,
                                             magnitude_thresh=magnitude_thresh, dir_thresh=dir_thresh)

        image_hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = image_hsl[:, :, 2]
        s_channel = cv2.medianBlur(s_channel, 5)
        color_threshold = np.zeros_like(s_channel)
        color_threshold[(s_channel >= parameters['sat_thresh_min']) & (s_channel <= parameters['sat_thresh_max'])] = 1

        binary = np.zeros_like(gradient_threshold)
        binary[(color_threshold == 1) | (gradient_threshold == 1)] = 1

        return binary

    def find_perspective_transform(self):
        image = self._image_undist
        imshape = image.shape

        src_points = [(600, 445), (675, 445), (1040, 685), (253, 685)]
        src = np.array([src_points], dtype=np.float32)
        dst = np.array([(350, 0), (950, 0), (950, imshape[0]), (350, imshape[0])], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        print('Writing perspective matrices...')
        with open('{}.pkl'.format(PERSPECTIVE_FILE), 'wb') as f:
            pickle.dump([M, M_inv], f)

        warped = cv2.warpPerspective(self._image_undist, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

        if self._verbose:
            Pipeline.save_image('warped_straight.jpg', warped)
            plt.imshow(warped)
            plt.waitforbuttonpress()

        print('Perspective matrices computed and stored')

    @staticmethod
    def load_perspective_transform():
        with open('{}.pkl'.format(PERSPECTIVE_FILE), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_image(fname, img, out_dir=IMAGES_DIR, **kwargs):
        mpimg.imsave(os.path.join(out_dir, fname), img, **kwargs)

    def run(self):
        perspective_file = '{}.pkl'.format(PERSPECTIVE_FILE)
        if not os.path.exists(perspective_file):
            print("Please run with '--perspective' to compute perspective matrices")
            return

        M, M_inv = Pipeline.load_perspective_transform()

        image = self._image_undist
        imshape = image.shape

        binary = self.apply_thresholing()

        if self._verbose:
            Pipeline.save_image('{}_binary.jpg'.format(self._file_name_no_ext), binary, cmap='gray')
            show_image_gray(binary)
            plt.waitforbuttonpress()

        print('Warping the image')
        warped = cv2.warpPerspective(self._image_undist, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

        if self._verbose:
            Pipeline.save_image('{}_warp.jpg'.format(self._file_name_no_ext), warped)

        binary_warped = cv2.warpPerspective(binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        if self._verbose:
            Pipeline.save_image('{}_binary_wap.jpg'.format(self._file_name_no_ext), binary_warped, cmap='gray')

        lane_finder = ConvolutionLaneFinder(binary_warped, self._image_undist, (M, M_inv), self._verbose, self._file_name_no_ext)
        result, left_curverad, right_curverad = lane_finder.find_lanes()

        print('Creating output image')

        Pipeline.save_image("{}.jpg".format(self._file_name_no_ext), result, out_dir=OUT_DIR)
        plt.imshow(result)
        plt.waitforbuttonpress()


def main(args):
    pipeline = Pipeline(args)

    if args.perspective:
        pipeline.find_perspective_transform()
    else:
        pipeline.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Path to the file for processing")
    parser.add_argument("--mode", choices=['image', 'video'], help="Mode to operate in (image or video)")
    parser.add_argument("--perspective", action="store_true", help="Find perspective matrices")
    parser.add_argument("--verbose", action="store_true", help="Enable verbosity")

    args = parser.parse_args()

    main(args)