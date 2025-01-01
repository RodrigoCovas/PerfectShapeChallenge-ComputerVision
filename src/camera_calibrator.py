from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os
from os.path import dirname, join

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def show_image(img: np.array, title: str = 'Image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(img: np.array, path: str):
    cv2.imwrite(path, img)

def get_chessboard_points(chessboard_shape, dx, dy):
    points = []
    for y in range(chessboard_shape[1]):
        for x in range(chessboard_shape[0]):
            points.append([x*dx, y*dy, 0])
    return np.array(points, np.float32)

if __name__ == "__main__":
    imgs_path= []
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    folder = os.path.join(current_directory,"data","Chessboard")
    folder = folder.replace("\\", "/") + "/"
    for filename in glob.glob(folder+ "*"):
        imgs_path.append(filename)
    imgs= load_images(imgs_path)
    corners= [cv2.findChessboardCorners(img, (7, 7)) for img in imgs]
    # If an image does not have corners, remove it from the list
    imgs= [img for img, corner in zip(imgs, corners) if corner[0]]
    corners= [cv2.findChessboardCorners(img, (7,7)) for img in imgs]
    imgs_gray= [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs]
    corners_copy= copy.deepcopy(corners)
    criteria= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    # TODO To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
    imgs_gray= [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    corners_refined= [cv2.cornerSubPix(i, cor[1], (7,7), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy= copy.deepcopy(imgs)

    for i,cor in enumerate(corners_refined):
        cv2.drawChessboardCorners(imgs_copy[i], (7,7), cor, True)

    # TODO Show images and save when needed
    imgs_with_points= [cv2.drawChessboardCorners(img, (7,7), corner[1], corner[0]) for img, corner in zip(imgs_copy, corners)]
    imgs_with_points_r= [cv2.drawChessboardCorners(img, (7,7), cor, True) for img, cor in zip(imgs_copy, corners_refined)]

    for i, img in enumerate(imgs_with_points):
        show_image(img, f'Image {i}')
        write_image(img, f'{folder}output_image_{i}.jpg')

    for i, img in enumerate(imgs_with_points_r):
        show_image(img, f'Image {i}')
        write_image(img, f'{folder}output_image_r_{i}.jpg')

    chessboard_points= [get_chessboard_points((7,7), 30, 30) for _ in corners if corners[0]]

    # Filter data and get only those with adequate detections
    valid_corners= [cor[1] for cor in corners if cor[0]]
    # Convert list to numpy array
    valid_corners= np.asarray(valid_corners, dtype=np.float32)

    # TODO
    rms, intrinsics, dist_coeffs, rvecs, tvecs= cv2.calibrateCamera(chessboard_points, valid_corners, imgs_gray[0].shape[::-1], None, None)

    # Obtain extrinsics
    extrinsics= list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    # Print outputs
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)