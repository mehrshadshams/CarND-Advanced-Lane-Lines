## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration_corners]: ./images/camera_cal/corners_found13.jpg "Camera Calibration Corners"
[calibration_1]: ./camera_cal/calibration1.jpg "Camera Calibration 1"
[calibration_undist]: ./images/camera_cal/calibration1_undistorted.jpg "Camera Calibration Undistorted"
[image1]: ./test_images/straight_lines1.jpg "Road"
[image1_undistorted]: ./images/straight_lines1_undist.jpg "Road Transformed"
[image1_warped]: ./images/warped_straight.jpg "Road Warped"
[ui]: ./images/ui.png "Thresholding UI"

[test1_binary]: ./images/test1_binary.jpg "Binary"
[test1_binary_warped]: ./images/test1_binary_wap.jpg "Binary warped"
[test1_binary_warped_fit]: ./images/test1_binary_warp_fit.jpg "Binary warped fit"
[test1_result]: ./output_images/test1.jpg "Result"
[straight1_binary]: ./images/straight_lines1_binary.jpg "Binary"
[straight1_binary_warped]: ./images/straight_lines1_binary_wap.jpg "Binary warped"
[straight1_binary_warped_fit]: ./images/straight_lines1_binary_warp_fit.jpg "Binary warped fit"
[straight1_result]: ./output_images/straight_lines1.jpg "Result"
[test5_binary]: ./images/test5_binary.jpg "Binary"
[test5_binary_warped]: ./images/test5_binary_wap.jpg "Binary warped"
[test5_binary_warped_fit]: ./images/test5_binary_warp_fit.jpg "Binary warped fit"
[test5_result]: ./output_images/test5.jpg "Result"

[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Camera Calibration

The first step is to calibrate camera used on the car using a set of checker board images provided. 
The code for this can be found in `camera_calibration.py`. As discussed in the lecture, I used `cv2::findChessboardCorners` 
on all of the images to get a mapping between corners found and object points, then I used `cv2::calibrateCamera` to find 
**camera matrix** and **distortion coefficient** which I store in a pickle file to be used later.

|            Corners                 |           Original         |         Undistorted             |
| ---------------------------------- |:--------------------------:| ------------------------------ :|
| ![alt text][calibration_corners]   | ![alt text][calibration_1] | ![alt text][calibration_undist] |

### Pipeline

The pipeline to find lane lines is defined in `pipeline.py`. You can see the options by running the following command:

```bash
python pipeline.py --help

usage: pipeline.py [-h] [--mode {image,video}] [--perspective] [--verbose]
                   [--skip_overlay] [--show]
                   filename

positional arguments:
  filename              Path to the file for processing

optional arguments:
  -h, --help            show this help message and exit
  --mode {image,video}  Mode to operate in (image or video)
  --perspective         Find perspective matrices
  --verbose             Enable verbosity
  --skip_overlay        Skip overlay
  --show                Enable showing results
```

#### Perspective Transform Matrix

Once we have the camera parameters, the first step is to find the **perspsective matrices** 
To do that I find 4 set of points from the following image and map that to four points in the destination file and 
use `getPerspectiveTransform` to get the perspective matrix and it's inverse. Then I store these matrices for later use in 
 a pickle file.

The code for my perspective transform includes a function called `find_perspective_transform()`, which appears in lines 
99 through 129 in the file `pipeline.py`:

```python
src_points = [(520, 504), (769, 504), (1100, imshape[0]), (217, imshape[0])]
src = np.array([src_points], dtype=np.float32)
dst = np.array([(350, 0), (950, 0), (950, imshape[0]), (350, imshape[0])], dtype=np.float32)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 520, 504      | 350, 0        | 
| 769, 504      | 950, 0        |
| 1100, 720     | 950, 720      |
| 217, 720      | 350, 720      |

And here's an example of perspective transform applied to an image

|            Original                 |           Undistorted         |         Bird eye view             |
| ---------------------------------- |:--------------------------:| ------------------------------ |
| ![alt text][image1]   | ![alt text][image1_undistorted] | ![alt text][image1_warped] |

#### Color thresholding

After I un-distort the image, I threshold it to extract a binary image showing only lane lines. 
To do this, as described in the lecture, we can use thresholding on a combination of variants of the image, including 
image gradient, gradient magnitude and direction, RGB and HSL value. To help me find an easier way to tweak these values
I implemented a simple UI like the following image:

![alt text][ui]

To achieve this result I applied the following transformations.

* Gradient in x and y direction using Sobel operator in method `(abs_sobel_thresh)`
* Gradient magnitude threshold in method `(mag_thresh)`
* Gradient direction threshold in method `(dir_threshold)`
* RGB thresholding to extract yellow color in method `(create_binary_image)`
* RGB thresholding to extract white color
* HLS thresholding to extract yellow color

I combined the result of these values into the binary image up above. The code for the thresholding can be 
 found in `utils.py` in the methods listed above.
 
#### Warping the image

The next step is to warp the binary image using the perspective matrix. 
```python
binary_warped = cv2.warpPerspective(binary, M, (imshape[1], imshape[0]), flags=cv2.INTER_LINEAR)
```
(`pipeline.py` line 119)

Here are some examples:

|            Binary           |           Binary warped          |
| --------------------------- |:--------------------------------:|
| ![alt text][straight1_binary]   | ![alt text][straight1_binary_warped] |
| ![alt text][test1_binary]   | ![alt text][test1_binary_warped] |
| ![alt text][test5_binary]   | ![alt text][test5_binary_warped] |

#### Curve fitting
 
The code for identifying lane pixels and fitting a polynomial curve to find the lines, 
can be found in `find_lanes.py`. I wrote a class called `LaneFitter` which can be used 
to fit the lines using `fit` method. This method is an improved version of the provided code. 
It uses sliding window over the image and uses the sum of the histogram of pixel values to find the lane lines.
As the sliding window moves over the binary image, we extract the indices of the non-zero pixels, 
then we can fit a second order polynomial on these indices (i.e. pixel locations) to find the 
coefficients for the left and right lines. (`find_lanes.py` lines 153 - 190)

Then I perform some sanity checking which mostly is used in the video mode. I check the slope of both lines
and reject the lines that don't have a similar slope. To do the compute the difference in slope I used the following
code, i.e. I compute slope for each line and then use MSE to compute the difference between those values.

```python
def _compute_slope_mse(self, ploty, fit):
    left_fit, right_fit = fit
    left_slope = 2 * ploty * left_fit[0] + left_fit[1]
    right_slope = 2 * ploty * right_fit[0] + right_fit[1]
    mse = np.sum(np.power(left_slope - right_slope, 2)) / len(left_slope)
    return mse
```

Then I ignore the lines if the `slope_diff` is greater than 0.1 (`find_lanes.py` lines 234-239)

```python
# do sanity checking, are two lines relatively parallel?
if left_fit is not None and right_fit is not None:
    slope_diff = self._compute_slope_mse(ploty, (left_fit, right_fit))
    if slope_diff > 0.1:
        print('ignoring fit')
        # ignore this fit
        left_fit, right_fit = None, None
```

The following images show few examples of curve fit on the binary image:

|            Binary           |           Binary warped          |
| --------------------------- |:--------------------------------:|
| ![alt text][straight1_binary_warped]   | ![alt text][straight1_binary_warped_fit] |
| ![alt text][test1_binary_warped]   | ![alt text][test1_binary_warped_fit] |
| ![alt text][test5_binary_warped]   | ![alt text][test5_binary_warped_fit] |

#### Computing Curvature

Once I have the polynomials it's possible to compute the curvature of the line by computing the radius of an approximate circle
at a given point on the line. This is done by computing the first and second derivatives of the formula of the line. 
The code below which is part of method `_compute_curvature` in `find_lanes.py` lines 95-109 does that.
 
```python
curve_rad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
```

To compute the vehicle's distance to the center line I use the following code: (`find_lanes.py` line 256)

```python
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
offset = xm_per_pix * ((left_base + (right_base - left_base) / 2) - imshape[1]//2)
```

> To compute the offset from the center, I use the left_base and right_base from the current frame

#### Result

Once I have all of the pieces I can plot the lines back to the original image using the 
 inverse perspective matrix we computed earlier.

![alt text][straight1_result]

![alt text][test1_result]

![alt text][test5_result]

### Pipeline (video)

The video pipeline uses the same pipeline as images with some minor improvements.
Since there is a chance that we can't find a good fit for some of the frames, I keep the last
5 frames using a ring buffer. I keep the base line and the fit parameters for each of the lines,
in an instance of a `Line` object. Having this, for each frame I compute a moving average
 of the fit parameters for the previous frames as shown in the following code: (`find_lanes.py` lines 77-93)
 
```python
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
```

To compute the moving average I used a normalized range of numbers in logarithmic space. 
I also compute the average of the position of baselines using the previous frames. This helped a lot
with the stability of the pipeline.

```python
def _compute_average_base(self):
    items = self._buffer.get()
    if len(items) > 0:
        left_base = list(map(lambda x: x[0].base, items))
        right_base = list(map(lambda x: x[1].base, items))

        left_base = int(np.average(left_base))
        right_base = int(np.average(right_base))

        return left_base, right_base

    return None, None
```

Since I keep track of the lines across frames once I find the baseline, for the next frames there's no need
 to search the whole image to find the line pixels, instead I can use the baseline computed from the last frame.
  
Here are the links to my videos:

* [Project Video](https://youtu.be/74GM8ggdKaY)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=74GM8ggdKaY
" target="_blank"><img src="https://img.youtube.com/vi/74GM8ggdKaY/0.jpg" 
alt="Udacity SDC Advanced Lane finding project video" width="240" height="180" border="10" /></a>

* [Challenge Video](https://youtu.be/BrsE81f5-CI)

<a href="http://www.youtube.com/watch?feature=player_embedded&v=BrsE81f5-CI
" target="_blank"><img src="https://img.youtube.com/vi/BrsE81f5-CI/0.jpg" 
alt="Udacity SDC Advanced Lane finding challenge video" width="240" height="180" border="10" /></a>

### Discussion

I was able to get good results on the `project_video` and `challenge_video` but the result of the `harder_challenge_video` 
were far from good. Partly because of the fact that the algorithm is not robust when thresholding shadowy areas. This causes many 
unwanted artifacts. Also fitting a polynomial may not be the best approach in this case. I would to try RANSAC to find the 
matching line because unlike polynomial fitting, RANSAC is actually robust against outliers, i.e. in the case that there 
are many outliers the resulted fit does not diverge from the previous frames that much.
 