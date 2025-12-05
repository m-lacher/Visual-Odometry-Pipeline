import os
import cv2
import numpy as np
from glob import glob
from src.initialization import process_frame, match_points
from src.visualizations import *
from src.helpers.map_points import MapPoint, calculate_essential_matrix_and_triangulate_map_points
from src.helpers.match_descriptors import matchDescriptorsLOWE

# code was adjusted from given main.py

def load_dataset(ds):
    
    # dataset paths
    kitti_path = "./Datasets/kitti"
    malaga_path = "./Datasets/malaga"
    parking_path = "./Datasets/parking"
    own_dataset_path = "./Datasets/own_dataset"

    K = None
    ground_truth = None
    last_frame = 0
    path_handle = None

    if ds == 0:
        ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
        last_frame = 4540
        K = np.array([
            [7.18856e+02, 0, 6.071928e+02],
            [0, 7.18856e+02, 1.852157e+02],
            [0, 0, 1]
        ])
        path_handle = kitti_path

    elif ds == 1:
        # Malaga
        img_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
        left_images = sorted(glob(os.path.join(img_dir, '*.jpg')))
        last_frame = len(left_images)
        K = np.array([
            [621.18428, 0, 404.0076],
            [0, 621.18428, 309.05989],
            [0, 0, 1]
        ])
        path_handle = left_images

    elif ds == 2:
        # Parking
        last_frame = 598
        K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=',', usecols=(0, 1, 2))
        ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
        path_handle = parking_path

    elif ds == 3:
        # our Dataset
        path_handle = own_dataset_path
        
    else:
        raise ValueError("Invalid dataset index")
        
    return K, ground_truth, last_frame, path_handle

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
    visualize_matches(p0, p1, img0, img1, max_matches=15)
    map_points = calculate_essential_matrix_and_triangulate_map_points(p0, p1, p1_descriptors, K, None) # store descriptors from later frame (more robust)
    return map_points

def continuous_operation(ds, path_handle, last_frame, start_index, map_points):

    viewer = WorldViewer2D(scale=3) # for visualization
    map_points_3d = np.array([mp.position for mp in map_points]).T
    viewer.add_points(map_points_3d.T)    # New landmark points from this frame
    prev_img = None
    
    for i in range(start_index, last_frame + 1):
        print(f"\nProcessing frame {i}")
        print("=====================")
        
        image_path = ""
        
        if ds == 0:
            image_path = os.path.join(path_handle, '05', 'image_0', f"{i:06d}.png")
        elif ds == 1:
            image_path = path_handle[i]
        elif ds == 2:
            image_path = os.path.join(path_handle, 'images', f"img_{i:05d}.png")
        elif ds == 3:
            image_path = os.path.join(path_handle, f"{i:06d}.png")
        else:
            raise ValueError("Invalid dataset index")
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        # describe new image features
        key_points, described_points = process_frame(img=image)
        # find new matches from existing landmarks
        map_descriptors = np.array([mp.descriptor for mp in map_points]).T
        print(map_points_3d.shape)
        matches = matchDescriptorsLOWE(described_points, map_descriptors, match_lambda=0.7)

        query_indices = np.nonzero(matches >= 0)[0]
        match_indices = matches[query_indices].astype(int)

        points_matched_2d = key_points[:, query_indices]       # 2D points in current image
        points_matched_3d = map_points_3d[:, match_indices]   # corresponding 3D map points

        points_matched_2d = points_matched_2d[::-1, :].T   # shape (N,2)
        points_matched_3d = points_matched_3d.T           # shape (N,3)
        
        # do PnP with Ransac
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(points_matched_3d),
            np.array(points_matched_2d),
            K,
            None
        )
        print(f"PnP Success: {success}")
        # If pnp was successful: Plot new camera pose
        if(success):
            print(f"Number of inliers found: {len(inliers)}")
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)

            R_cw = R.T          # camera rotation in world coordinates
            t_cw = -R_cw @ t    # camera position in world coordinates

            print(t_cw)
            viewer.add_camera(R_cw, t_cw)         # add new camera pose
            viewer.draw()                         # Refresh display
        
        
        prev_img = image
        # Simulate 'pause(0.01)'
        cv2.waitKey(10)


if __name__ == "__main__":

    # 0 Initial config
    ds = 2  # 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
    bootstrap_frames = [0, 4]   # which two images to use for finding initial landmarks
    
    # 1 Load Dataset
    K, ground_truth, last_frame, path_handle = load_dataset(ds)
    print(f"Dataset loaded. Last frame: {last_frame}")
    print(f"K matrix: \n{K}")
    
    # 2 Initialization
    map_points = initialize(ds, path_handle, bootstrap_frames)
    
    #visualize_matches_zoomed(p0, p1, img0, img1, zoom_radius=30, scale_factor=6, max_matches=10)

    # 3. Continuous Operation
    start_index = bootstrap_frames[1] + 1
    continuous_operation(ds, path_handle, last_frame, start_index, map_points)