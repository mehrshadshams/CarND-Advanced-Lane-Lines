import argparse
import cv2
import numpy as np
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import os
import logging
import traceback
from typing import List

from utils import abs_sobel_thresh, mag_thresh, dir_threshold, undistort_image, show_image_gray, create_binary_image
from find_lanes import LaneFitter, Curvature

from moviepy.editor import VideoFileClip

OUT_DIR = "output_images"
IMAGES_DIR = "images"
PERSPECTIVE_FILE = "perspective"


class Parameters(object):
    def __init__(self, frame, left_curvature: Curvature, right_curvature: Curvature, offset):
        self.frame = frame
        self.left_curvature = left_curvature
        self.right_curvature = right_curvature
        self.offset = offset


class Pipeline(object):
    def __init__(self, args):
        self._filename = args.filename
        self._verbose = args.verbose
        self._show = args.show
        self._filename_no_ext = os.path.splitext(os.path.split(args.filename)[-1])[0]
        self._frame = 0
        self._parameters = Parameters(0, 0, 0, 0)
        self._add_overlay = not args.skip_overlay

        if self._verbose and not os.path.exists(IMAGES_DIR):
            os.mkdir(IMAGES_DIR)

        logging.info('Loading threshold parameters...')
        with open('parameters.json', 'r') as f:
            self._thresh_params = json.load(f)['adjusted']

        logging.info('Loading camera calibration parameters...')
        with open('camera.pkl', 'rb') as f:
            camera = pickle.load(f)

        self._mtx, self._dist = camera['mtx'], camera['dist']

        if not args.perspective:
            result = Pipeline.load_perspective_transform()
            if result is None:
                logging.warning("Please run with '--perspective' to compute perspective matrices")
            else:
                self._M, self._M_inv = result
                self._lane_fitter: LaneFitter = LaneFitter((self._M, self._M_inv), self._verbose)

    def apply_thresholing(self, image):
        parameters = self._thresh_params

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # image_gray = clahe.apply(image_gray)

        gradx_thresh = (parameters['sobel_x_thresh_min'], parameters['sobel_x_thresh_max'])
        grady_thresh = (parameters['sobel_y_thresh_min'], parameters['sobel_y_thresh_max'])
        magnitude_thresh = (parameters['mag_thresh_min'], parameters['mag_thresh_max'])
        dir_thresh = (parameters['dir_thresh_min'], parameters['dir_thresh_max'])

        ksize = parameters['sobel_kernel']

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(image_gray, orient='x', sobel_kernel=ksize, thresh=gradx_thresh)
        grady = abs_sobel_thresh(image_gray, orient='y', sobel_kernel=ksize, thresh=grady_thresh)
        mag_binary = mag_thresh(image_gray, sobel_kernel=ksize, thresh=magnitude_thresh)
        dir_binary = dir_threshold(image_gray, sobel_kernel=ksize, thresh=dir_thresh)

        gradient_threshold = np.zeros_like(dir_binary)
        gradient_threshold[((gradx == 1) & (grady == 1)) | ((dir_binary == 1) & (mag_binary == 1))] = 1

        color_threshold = create_binary_image(image)

        binary = np.zeros_like(gradient_threshold)
        binary[(color_threshold == 1) | (gradient_threshold == 1)] = 1
        # binary[(gradient_threshold == 1)] = 1

        return binary

    def load_image(self):
        image = cv2.imread(self._filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def find_perspective_transform(self):
        image = undistort_image(self.load_image(), self._mtx, self._dist)

        Pipeline.save_image('{}_undistorted.jpg'.format(self._filename_no_ext), image)

        imshape = image.shape

        src_points = [(520, 504), (769, 504), (1100, imshape[0]), (217, imshape[0])]
        src = np.array([src_points], dtype=np.float32)
        dst = np.array([(350, 0), (950, 0), (950, imshape[0]), (350, imshape[0])], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        logging.info('Writing perspective matrices...')
        with open('{}.pkl'.format(PERSPECTIVE_FILE), 'wb') as f:
            pickle.dump([M, M_inv], f)

        # im2 = cv2.polylines(image, src.astype(np.int32), 1, (0, 255, 0), thickness=2)

        warped = cv2.warpPerspective(image, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

        Pipeline.save_image('warped_straight.jpg', warped)

        if self._verbose:
            # warped = cv2.polylines(warped, dst.astype(np.int32)[:, np.newaxis], 1, (0, 255, 0), thickness=2)

            plt.imshow(warped)
            plt.waitforbuttonpress()

        logging.info('Perspective matrices computed and stored')

    @staticmethod
    def load_perspective_transform():
        perspective_file = '{}.pkl'.format(PERSPECTIVE_FILE)
        if not os.path.exists(perspective_file):
            return None

        with open('{}.pkl'.format(PERSPECTIVE_FILE), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_image(fname, img, out_dir=IMAGES_DIR, **kwargs):
        mpimg.imsave(os.path.join(out_dir, fname), img, **kwargs)

    @staticmethod
    def factory(args):
        if args.mode == 'image':
            return ImagePipeline(args)
        elif args.mode == 'video':
            return VideoPipeline(args)

        raise RuntimeError("Unknown mode {}".format(args.mode))

    @staticmethod
    def add_overlay(result, parameters: Parameters, *args):
        if args is None or len(args) == 0:
            return result

        imshape = result.shape

        coeff = max(4, len(args))
        temp_shape = (imshape[0] // coeff, imshape[1] // coeff)
        append = np.zeros((imshape[0], temp_shape[1], 3), dtype=np.uint8)

        for idx, img in enumerate(args):
            temp = cv2.resize(img, (temp_shape[1], temp_shape[0]))

            if len(temp.shape) < 3:
                temp = np.dstack([temp, temp, temp])

            temp = (temp / temp.max() * 255).astype(np.uint8)

            append[idx * temp.shape[0]:(idx + 1) * temp.shape[0], :temp.shape[1], :] = temp

        text = 'Left = {}m, Right = {}m, Offset = {:.2}m'\
            .format(int(parameters.left_curvature.world), int(parameters.right_curvature.world), parameters.offset)
        result = cv2.putText(img=result, text=text, org=(0, 50),
                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))

        text = '{0:05d}'.format(parameters.frame)
        result = cv2.putText(img=result, text=text, org=(imshape[1] - 100, 50),
                             fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(255, 255, 255))

        return np.hstack([result, append])

    def run(self):
        pass

    def process_image(self, image):
        self._frame += 1

        M, M_inv = self._M, self._M_inv

        image_undist = undistort_image(image, self._mtx, self._dist)

        if self._verbose:
            Pipeline.save_image('{}_undist.jpg'.format(self._filename_no_ext), image_undist)

        imshape = image.shape

        logging.info('Creating binary threshold...')
        binary = self.apply_thresholing(image_undist)

        if self._verbose:
            Pipeline.save_image('{}_binary.jpg'.format(self._filename_no_ext), binary, cmap='gray')

        if self._show:
            show_image_gray(binary)
            plt.waitforbuttonpress()

        logging.info('Warping the image')
        warped = cv2.warpPerspective(image_undist, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)

        if self._verbose:
            logging.info('Warped image')
            Pipeline.save_image('{}_warp.jpg'.format(self._filename_no_ext), warped)

        binary_warped = cv2.warpPerspective(binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
        if self._verbose:
            Pipeline.save_image('{}_binary_wap.jpg'.format(self._filename_no_ext), binary_warped, cmap='gray')

        result, fit_image, curvatures, offset = self._lane_fitter.fit_transform(binary_warped, image_undist)

        if self._verbose:
            Pipeline.save_image('{}_binary_warp_fit.jpg'.format(self._filename_no_ext), fit_image)

        p = self._parameters
        if self._frame - p.frame >= 5 or p.frame == 0:
            self._parameters = Parameters(self._frame, curvatures[0], curvatures[1], offset)
            p = self._parameters

        if self._add_overlay:
            result = Pipeline.add_overlay(result, p, binary, warped, fit_image)

        logging.info('Creating output image')

        if self._verbose:
            Pipeline.save_image("{}.jpg".format(self._filename_no_ext), result, out_dir=OUT_DIR)

        if self._show:
            plt.imshow(result)
            plt.waitforbuttonpress()

        return result


class ImagePipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        logging.info("Loading '{}'".format(args.filename))
        image = cv2.imread(args.filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._image = image

    def run(self):
        self.process_image(self._image)


class VideoPipeline(Pipeline):
    def __init__(self, args):
        super().__init__(args)
        self._clip = VideoFileClip(self._filename)

    def process_image(self, image):
        try:
            return super().process_image(image)
        except Exception as e:
            traceback.print_exc()
            return image

    def run(self):
        output = os.path.join(OUT_DIR, self._filename)
        white_clip = self._clip.fl_image(self.process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output, audio=False)


def main(args):
    if not os.path.exists(args.filename):
        print('File {} not found.'.format(args.filename))
        return

    pipeline = Pipeline.factory(args)

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
    parser.add_argument("--skip_overlay", action="store_true", default=False, help="Skip overlay")
    parser.add_argument("--show", action="store_true", help="Enable showing results")

    args = parser.parse_args()

    if args.mode != 'video':
        logging.basicConfig(level=logging.INFO)

    main(args)
