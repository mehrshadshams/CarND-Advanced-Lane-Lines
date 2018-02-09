import cv2
import glob
import numpy as np
import pickle
import os
from utils import undistort_image


def main():
    images = glob.glob('camera_cal/*.jpg')

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    if not os.path.exists('images/camera_cal'):
        os.mkdir('images/camera_cal')

    for idx,fname in enumerate(images):
        img = cv2.imread(fname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img_gray, (9, 6), None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            write_name = 'images/camera_cal/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (720, 1280), None, None)

    print('Writing camera matrix and distortion coefficients...')
    with open('camera.pkl', 'wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)

    undistorted = undistort_image(cv2.imread('camera_cal/calibration1.jpg'), mtx, dist)
    cv2.imwrite("images/camera_cal/calibration1_undistorted.jpg", undistorted)


if __name__ == "__main__":
    main()
