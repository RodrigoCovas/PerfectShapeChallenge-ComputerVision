import cv2
import os
import numpy as np
import imageio
from typing import List
import glob
import copy

def grade_circle(image_path, num_sample_points=100, max_tolerable_deviation=10):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image could not be loaded. Check the path.")

    # Threshold the image to create a binary version
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Assume the largest contour is the sketched circle
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a minimum enclosing circle to the largest contour
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)

    # Reduce the number of points for averaging
    contour_points = largest_contour[:, 0, :]  # Extract (x, y) points
    step = max(1, len(contour_points) // num_sample_points)  # Step size to sample points
    sampled_points = contour_points[::step]

    # Calculate the average deviation of sampled points from the best-fit circle
    distances = []
    for px, py in sampled_points:
        distance_to_circle = abs(np.sqrt((px - center_x)**2 + (py - center_y)**2) - radius)
        distances.append(distance_to_circle)
    
    avg_deviation = np.mean(distances)

    # Normalize the score (100% for perfect circle, lower for higher deviations)
    score = max(0, 100 - (avg_deviation / max_tolerable_deviation) * 100)
    score = round(score, 2)  # Round to 2 decimal places

    return score

# Test the function
if __name__ == "__main__":
    input_image_path = "C:/Users/rodri/Documents/3 IMAT/Cuatri 1/Computer Vision/CVI-ICAI/Lab_Project/data/sketch1.png"
    try:
        # score = grade_circle(input_image_path, 100, 14000)
        # score = grade_circle(input_image_path, 100, 10000)
        score = grade_circle(input_image_path, 100, 500)
        # score = grade_circle(input_image_path, 100, 11000)
        print(f"Circle Score: {score}%")
    except FileNotFoundError as e:
        print(str(e))
