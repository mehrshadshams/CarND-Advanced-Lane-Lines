import cv2
import numpy as np
import json
import matplotlib.image as mpimg

from pyforms import BaseWidget
from pyforms.Controls import ControlButton
from pyforms.Controls import ControlSlider
from pyforms.Controls import ControlImage
from utils import abs_sobel_thresh, threshold_image, mag_thresh, dir_threshold, threshold_color
from color_test import thresh_bin_im

import pyforms


class ThresholdingParameter(BaseWidget):
    def __init__(self):
        super(ThresholdingParameter, self).__init__('Computer vision algorithm example')

        with open('parameters.json', 'r') as f:
            params_values = json.load(f)

        self._params_values = params_values

        params = params_values['adjusted']

        # Definition of the forms fields
        self.sobel_kernel = ControlSlider('sobel_kernel', minimum=1, maximum=15, default=int(params['sobel_kernel']))

        self.sobel_x_thresh_min = ControlSlider('sobel_x_thresh_min', minimum=1, maximum=255, default=int(params['sobel_x_thresh_min']))
        self.sobel_x_thresh_max = ControlSlider('sobel_x_thresh_max', minimum=1, maximum=255, default=int(params['sobel_x_thresh_max']))
        self.sobel_y_thresh_min = ControlSlider('sobel_y_thresh_min', minimum=1, maximum=255, default=int(params['sobel_y_thresh_min']))
        self.sobel_y_thresh_max = ControlSlider('sobel_y_thresh_max', minimum=1, maximum=255, default=int(params['sobel_y_thresh_max']))

        self.mag_kernel = ControlSlider('mag_kernel', minimum=1, maximum=15, default=int(params['mag_kernel']))
        self.mag_thresh_min = ControlSlider('mag_thresh_min', minimum=1, maximum=255, default=int(params['mag_thresh_min']))
        self.mag_thresh_max = ControlSlider('mag_thresh_max', minimum=1, maximum=255, default=int(params['mag_thresh_max']))

        self.dir_kernel = ControlSlider('dir_kernel', minimum=1, maximum=15, default=int(params['dir_kernel']))
        self.dir_thresh_min = ControlSlider('dir_thresh_min', minimum=0, maximum=int((np.pi / 2) * 10),
                                            default=int(float(params['dir_thresh_min']) * 10))
        self.dir_thresh_max = ControlSlider('dir_thresh_max', minimum=1, maximum=int((np.pi / 2) * 10),
                                            default=int(float(params['dir_thresh_max']) * 10))

        self.sat_thresh_min = ControlSlider('sat_thresh_min', minimum=1, maximum=255, default=int(params['sat_thresh_min']))
        self.sat_thresh_max = ControlSlider('sat_thresh_max', minimum=1, maximum=255, default=int(params['sat_thresh_max']))

        self._image = ControlImage()
        self._runbutton = ControlButton('Run')

        # Define the function that will be called when a file is selected
        self.sobel_kernel.changed_event = self.__sobel_kernelEvent
        self.sobel_x_thresh_min.changed_event = self.__sobel_x_thresh_minEvent
        self.sobel_x_thresh_max.changed_event = self.__sobel_x_thresh_maxEvent
        self.sobel_y_thresh_min.changed_event = self.__sobel_y_thresh_minEvent
        self.sobel_y_thresh_max.changed_event = self.__sobel_y_thresh_maxEvent
        self.mag_kernel.changed_event = self.__mag_kernelEvent
        self.mag_thresh_min.changed_event = self.__mag_thresh_minEvent
        self.mag_thresh_max.changed_event = self.__mag_thresh_maxEvent
        self.dir_kernel.changed_event = self.__dir_kernelEvent
        self.dir_thresh_min.changed_event = self.__dir_thresh_minEvent
        self.dir_thresh_max.changed_event = self.__dir_thresh_maxEvent
        self.sat_thresh_min.changed_event = self.__sat_thresh_minEvent
        self.sat_thresh_max.changed_event = self.__sat_thresh_maxEvent

        # Define the event that will be called when the run button is processed
        self._runbutton.value = self.__runEvent

        self.input_image = cv2.imread('test_images/test1.jpg')
        self.input_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite("a.jpg", self.input_image)

        self._binary = None

        # Define the event called before showing the image in the player
        # self._player.processFrame = self.__processFrame

        # Define the organization of the Form Controls
        self.formset = [
            ('sobel_kernel',
            'sobel_x_thresh_min',
            'sobel_x_thresh_max'),
            ('sobel_y_thresh_min',
            'sobel_y_thresh_max',
            'mag_kernel'),
            ('mag_thresh_min',
            'mag_thresh_max',
            'dir_kernel'),
            ('dir_thresh_min',
            'dir_thresh_max',
            'sat_thresh_min',
            'sat_thresh_max'), '_image',
            ('_runbutton')
        ]

    def init_form(self, parse=True):
        super(ThresholdingParameter, self).init_form()

        self.process_video()

    def __sobel_kernelEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        if self.sobel_kernel.value % 2 == 1:
            self.process_video()

    def __sobel_x_thresh_minEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __sobel_x_thresh_maxEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __sobel_y_thresh_minEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __sobel_y_thresh_maxEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __mag_kernelEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        if self.mag_kernel.value % 2 == 1:
            self.process_video()

    def __mag_thresh_minEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __mag_thresh_maxEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __dir_kernelEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        if self.dir_kernel.value % 2 == 1:
            self.process_video()

    def __dir_thresh_minEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __dir_thresh_maxEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __sat_thresh_minEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def __sat_thresh_maxEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self.process_video()

    def _write_image(self, img):
        mpimg.imsave('out.jpg', img)

    def process_video(self):
        test_image = self.input_image

        sobel_kernel = self.sobel_kernel.value
        sobel_x_thresh_min = self.sobel_x_thresh_min.value
        sobel_x_thresh_max = self.sobel_x_thresh_max.value
        sobel_y_thresh_min = self.sobel_y_thresh_min.value
        sobel_y_thresh_max = self.sobel_y_thresh_max.value

        mag_kernel = self.mag_kernel.value
        mag_thresh_min = self.mag_thresh_min.value
        mag_thresh_max = self.mag_thresh_max.value
        dir_kernel = self.dir_kernel.value
        dir_thresh_min = self.dir_thresh_min.value / 10
        dir_thresh_max = self.dir_thresh_max.value / 10
        sat_thresh_min = self.sat_thresh_min.value
        sat_thresh_max = self.sat_thresh_max.value

        # gradient_threshold = threshold_image(test_image, ksize=sobel_kernel,
        #                                      gradx_thresh=(sobel_x_thresh_min, sobel_x_thresh_max),
        #                                      grady_thresh=(sobel_y_thresh_min, sobel_y_thresh_max),
        #                                      magnitude_thresh=(mag_thresh_min, mag_thresh_max),
        #                                      dir_thresh=(dir_thresh_min, dir_thresh_max))

        ksize = sobel_kernel

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(test_image, orient='x', sobel_kernel=ksize, thresh=(sobel_x_thresh_min, sobel_x_thresh_max))
        grady = abs_sobel_thresh(test_image, orient='y', sobel_kernel=ksize, thresh=(sobel_y_thresh_min, sobel_y_thresh_max))
        mag_binary = mag_thresh(test_image, sobel_kernel=ksize, thresh=(mag_thresh_min, mag_thresh_max))
        dir_binary = dir_threshold(test_image, sobel_kernel=ksize, thresh=(dir_thresh_min, dir_thresh_max))

        gradient_threshold = np.zeros_like(dir_binary)
        # gradient_threshold[(gradx == 1)] = 1
        gradient_threshold[((gradx == 1) & (grady == 1)) | ((dir_binary == 1) & (mag_binary == 1))] = 1

        image_hsl = cv2.cvtColor(test_image, cv2.COLOR_RGB2HLS)
        s_channel = image_hsl[:, :, 2]
        color_threshold = np.zeros_like(s_channel)
        color_threshold[(s_channel >= sat_thresh_min) & (s_channel <= sat_thresh_max)] = 1

        color_threshold = thresh_bin_im(test_image)
        binary = np.zeros_like(gradient_threshold)
        binary[(color_threshold == 1) | (gradient_threshold == 1)] = 1

        # binary = np.zeros_like(gradient_threshold)
        # binary[(color_threshold == 1) | (gradient_threshold == 1)] = 1
        # binary[(gradient_threshold == 1)] = 1

        self._write_image(binary)

        self._binary = cv2.imread('out.jpg', 0)

        print('Created image')

        self._image.value = self._binary
        self._image.repaint()

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        sobel_kernel = self.sobel_kernel.value
        sobel_x_thresh_min = self.sobel_x_thresh_min.value
        sobel_x_thresh_max = self.sobel_x_thresh_max.value
        sobel_y_thresh_min = self.sobel_y_thresh_min.value
        sobel_y_thresh_max = self.sobel_y_thresh_max.value

        mag_kernel = self.mag_kernel.value
        mag_thresh_min = self.mag_thresh_min.value
        mag_thresh_max = self.mag_thresh_max.value
        dir_kernel = self.dir_kernel.value
        dir_thresh_min = self.dir_thresh_min.value / 10
        dir_thresh_max = self.dir_thresh_max.value / 10
        sat_thresh_min = self.sat_thresh_min.value
        sat_thresh_max = self.sat_thresh_max.value

        values = self._params_values
        values['adjusted'] = {
            "sobel_kernel": sobel_kernel,
            "mag_kernel": mag_kernel,
            "dir_kernel": dir_kernel,
            "sobel_x_thresh_min": sobel_x_thresh_min,
            "sobel_x_thresh_max": sobel_x_thresh_max,
            "sobel_y_thresh_min": sobel_y_thresh_min,
            "sobel_y_thresh_max": sobel_y_thresh_max,
            "mag_thresh_min": mag_thresh_min,
            "mag_thresh_max": mag_thresh_max,
            "dir_thresh_min": dir_thresh_min,
            "dir_thresh_max": dir_thresh_max,
            "sat_thresh_min": sat_thresh_min,
            "sat_thresh_max": sat_thresh_max
        }

        with open('parameters.json', 'w') as f:
            json.dump(values, f)

        self._params_values = values

        if self._binary is not None:
            cv2.imwrite('binary.jpg', self._binary)


if __name__ == "__main__":	 pyforms.start_app(ThresholdingParameter)