from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os
from os.path import dirname, join

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

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

imgs_path2 = []
folder2 = "" # Path para las imágenes de la cámara derecha
for filename in glob.glob(folder2 + "*.jpg"):
    imgs_path2.append(filename)
imgs2 = load_images(imgs_path2)

corners2 = [cv2.findChessboardCorners(img, (8, 6)) for img in imgs2]
# If an image does not have corners, remove it from the list
imgs2 = [img for img, corner in zip(imgs2, corners2) if corner[0]]
corners2 = [cv2.findChessboardCorners(img, (8, 6)) for img in imgs2]
imgs_gray2 = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgs2]

corners_copy2 = copy.deepcopy(corners2)
criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

# TODO To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.
imgs_gray2 = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs2]

corners_refined2 = [cv2.cornerSubPix(i, cor[1], (8, 6), (-1, -1), criteria2) if cor[0] else [] for i, cor in zip(imgs_gray2, corners_copy2)]

imgs_copy2 = copy.deepcopy(imgs2)

for i,cor in enumerate(corners_refined2):
    cv2.drawChessboardCorners(imgs_copy2[i], (8,6), cor, True)

# TODO Show images and save when needed
imgs_with_points2 = [cv2.drawChessboardCorners(img, (8, 6), corner[1], corner[0]) for img, corner in zip(imgs_copy2, corners2)]
imgs_with_points_r2 = [cv2.drawChessboardCorners(img, (8, 6), cor, True) for img, cor in zip(imgs_copy2, corners_refined2)]

for i, img in enumerate(imgs_with_points2):
    show_image(img, f'Image {i}')
    write_image(img, f'output_image2_{i}.jpg')

for i, img in enumerate(imgs_with_points_r2):
    show_image(img, f'Image {i}')
    write_image(img, f'output_image2_r_{i}.jpg')

chessboard_points2 = [get_chessboard_points((8, 6), 30, 30) for _ in corners2 if corners2[0]]

# Filter data and get only those with adequate detections
valid_corners2 = [cor[1] for cor in corners2 if cor[0]]
# Convert list to numpy array
valid_corners2 = np.asarray(valid_corners2, dtype=np.float32)

# TODO
rms2, intrinsics2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(chessboard_points2, valid_corners2, imgs_gray2[0].shape[::-1], None, None)

# Obtain extrinsics
extrinsics2 = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs2, tvecs2))

# Print outputs
print("Intrinsics:\n", intrinsics2)
print("Distortion coefficients:\n", dist_coeffs2)
print("Root mean squared reprojection error:\n", rms2)