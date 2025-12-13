import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from src.initialization import process_frame, match_points
from src.visualizations import *
from src.helpers.map_points import MapPoint, calculate_essential_matrix_and_triangulate_map_points
from src.helpers.match_descriptors import matchDescriptorsLOWE
from src.continuous_operation import continuous_operation
from src.dataset import load_dataset


def initialize(ds, path_handle, frame_indices):

    # Step 1 - loading initial images
    img0 = None
    img1 = None

    if ds == 0:
        img0 = cv2.imread(os.path.join(path_handle, '05', 'image_0', f"{frame_indices[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(path_handle, '05', 'image_0', f"{frame_indices[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    elif ds == 1:
        img0 = cv2.imread(path_handle[frame_indices[0]], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(path_handle[frame_indices[1]], cv2.IMREAD_GRAYSCALE)
    elif ds == 2:
        img0 = cv2.imread(os.path.join(path_handle, 'images', f"img_{frame_indices[0]:05d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(path_handle, 'images', f"img_{frame_indices[1]:05d}.png"), cv2.IMREAD_GRAYSCALE)
    elif ds == 3:
        img0 = cv2.imread(os.path.join(path_handle, f"{frame_indices[0]:06d}.png"), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(path_handle, f"{frame_indices[1]:06d}.png"), cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Invalid dataset index")
        
    # Step 2 - possible  setting model parameters for object matching (done in different files for now)
    #corner_patch_size = 9
    #harris_kappa = 0.08
    #num_keypoints = 200
    #nonmaximum_supression_radius = 8
    #descriptor_radius = 9
    #match_lambda = 4

    # Step 3 - processing the two frames
    key_points_0, described_points_0 = process_frame(img=img0)
    key_points_1, described_points_1 = process_frame(img=img1)

    #visualize_keypoints(key_points_0, img0, "Frame 0: All Keypoints")
    #visualize_keypoints(key_points_1, img1, "Frame 1: All Keypoints")

    # Step 4 - maching the points from the frames
    p0, p1, p0_descriptors, p1_descriptors = match_points(key_points_0, described_points_0, key_points_1, described_points_1, match_lambda=0.7)
    visualize_matches(p0, p1, img0, img1, max_matches=30)
    map_points = calculate_essential_matrix_and_triangulate_map_points(p0, p1, p1_descriptors, K, frame_indices, None) # store descriptors from later frame (more robust)
    return map_points


if __name__ == "__main__":

    # 0 Initial config
    ds = 1  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
    bootstrap_frames = [0, 4]   # which two images to use for finding initial landmarks
    
    # 1 Load Dataset
    K, ground_truth, last_frame, path_handle = load_dataset(ds)
    print(f"Dataset loaded. Last frame: {last_frame}")
    print(f"K matrix: \n{K}")
    
    # 2 Initialization
    map_points = initialize(ds, path_handle, bootstrap_frames)

    # 3. Continuous Operation
    start_index = bootstrap_frames[1] + 1
    map_points = continuous_operation(ds, path_handle, last_frame, start_index, map_points, K)
