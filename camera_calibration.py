import cv2
import glob
import numpy as np
import pickle


def main():
    images = glob.glob('camera_cal/*.jpg')

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    for fname in images:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(img, (9, 6), None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (720, 1280), None, None)

    print('Writing camera matrix and distortion coefficients...')
    with open('camera.pkl', 'wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)


if __name__ == "__main__":
    main()
