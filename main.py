import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from src.initialization import process_frame, match_points
from src.visualizations import *
from src.helpers.map_points import MapPoint, calculate_essential_matrix_and_triangulate_map_points
from src.helpers.match_descriptors import matchDescriptorsLOWE
from src.helpers.bundle_adjustment import bundle_adjustment
from src.helpers.ba_bridge import build_optimization_problem, update_objects_from_state
from src.helpers.runBA import runBA

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
    visualize_matches(p0, p1, img0, img1, max_matches=30)
    map_points = calculate_essential_matrix_and_triangulate_map_points(p0, p1, p1_descriptors, K, frame_indices, None) # store descriptors from later frame (more robust)
    return map_points

def continuous_operation(ds, path_handle, last_frame, start_index, map_points):

    keyframe_dist = 4 #defines every nth Frame is a keyframe
    last_keyframe = None #initializes last keyframe
    keyframe_history = []
    MAX_HISTORY = 6 #bundle window
    last_keyframe_idx = start_index

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
        
        is_keyframe = (i - start_index) % keyframe_dist == 0 #checks if the current image is a keyframe
        
        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        #display the mapped points
        map_points_3d = np.array([mp.position for mp in map_points]).T
        viewer.add_points(map_points_3d.T)    # New landmark points from this frame

        # describe new image features
        key_points, described_points = process_frame(img=image)
        # find new matches from existing landmarks
        map_descriptors = np.array([mp.descriptor for mp in map_points]).T
        if i == start_index: #first frame after initialization
            matches = matchDescriptorsLOWE(described_points, map_descriptors, match_lambda=0.7) #matches using the map descriptors
            query_indices = np.nonzero(matches >= 0)[0]
            match_indices = matches[query_indices].astype(int)
            points_matched_3d = map_points_3d[:, match_indices] #selects the matched 3d points from the map
        elif (i - start_index) % keyframe_dist == 1: #second frame after keyframe
            matches = matchDescriptorsLOWE(described_points, map_descriptors, match_lambda=0.7) #matches using the updated map descriptors from last keyframe
            query_indices = np.nonzero(matches >= 0)[0]
            match_indices = matches[query_indices].astype(int)
            points_matched_3d = map_points_3d[:, match_indices] #selects the matched 3d points from the map
        else: #all other frames
            matches = matchDescriptorsLOWE(described_points, prev_matched_dp, match_lambda=0.7) #matches using the previous frame descriptors (only valid points from the last frame matching)
            query_indices = np.nonzero(matches >= 0)[0]
            match_indices = matches[query_indices].astype(int)
            points_matched_3d = prev_matched_kp3d[:, match_indices] #select the matched 3d points from the previous frame matching

        

        points_matched_2d = key_points[:, query_indices]       # 2D points in current image
        descriptors_matched_2d = described_points[:, query_indices] #corresponding descriptors
        
        prev_matched_kp = points_matched_2d #saves keypoints for the next frame
        prev_matched_dp = descriptors_matched_2d #saves descriptors for the next frame
        prev_matched_kp3d = points_matched_3d #saves 3d points for the next frame

        points_matched_2d = points_matched_2d[::-1, :].T   # shape (N,2) for PnP
        points_matched_3d = points_matched_3d.T           # shape (N,3) for PnP
        
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

            R_wc = R.T          # camera rotation in world coordinates (Note Gian-Andrin: changed notation to match the usual convention)
            t_wc = -R_wc @ t    # camera position in world coordinates (Note Gian-Andrin: changed notation to match the usual convention)
            projection_matrix = np.hstack((K @ R, K @ t))

            #print(R_wc)
            #print(t_wc)
            viewer.add_camera(R_wc, t_wc)         # add new camera pose
            viewer.draw()                         # Refresh display

            for idx in inliers.ravel():
                map_idx = match_indices[idx] 
                mp = map_points[map_idx]
                uv = tuple(points_matched_2d[idx])
                mp.add_observation(i, uv)
        
        """
        Note Gian-Andrin: The following was my notation:
        kp - keypoints
        dp - described points
        lkf - last keyframe
        ckf - current keyframe
        pm - projection matrix
        """
        if is_keyframe and (i - start_index) != 0 and success: #checks if current frame is a keyframe, PnP was successful, and is not the first keyframe
            viewer.clear_points() #clears previous point cloud from viewer to avoid cluttering
            p_lkf, p_ckf, p_lkf_descriptors, p_ckf_descriptors = match_points(lkf_kp, lkf_dp, key_points, described_points, match_lambda=0.7) #matches between last keyframe and current keyframe
            if len(p_lkf) <8:
                print("Triangulation failed")
            else:
                F, mask = cv2.findEssentialMat(p_lkf,p_ckf, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0) #runs RANSAC to find inliers (fundamental matrix not used)
                if mask is None:
                    print("triangulation failed")
                else:
                    inlier_mask = mask.ravel().astype(bool)
                    p_lkf_inliers = p_lkf[inlier_mask] #selects only inliers
                    p_ckf_inliers = p_ckf[inlier_mask] #selects only inliers
                    inlier_descriptors = p_ckf_descriptors[:, inlier_mask] #selects only inliers

                    new_map_points = cv2.triangulatePoints(lkf_pm, projection_matrix, p_lkf_inliers.T, p_ckf_inliers.T) #triangulates new point cloud from inlier matches between keyframes
                    new_map_points_normalized = (new_map_points[:3] / new_map_points[3]).T #normalizes the points
                    new_map_points_normalized_CF = R @ new_map_points_normalized.T + t #transforms points to camera frame for filtering
                    
                    valid_point_mask_min = new_map_points_normalized_CF[2,:] > 0 #filters points that are behind the camera
                    valid_point_mask_max = new_map_points_normalized_CF[2,:] < 30 #filters points that are too far away
                    valid_point_mask = valid_point_mask_min & valid_point_mask_max #combines the masks

                    new_map_points_normalized = new_map_points_normalized[valid_point_mask] #applies mask and filters points
                    inlier_descriptors = inlier_descriptors[:, valid_point_mask] #applies mask to descriptors
                    new_map_points_list = [MapPoint(new_map_points_normalized[s], inlier_descriptors[:,s]) for s in range(len(new_map_points_normalized))] #creates new map points list

                    for idx, mp in enumerate(new_map_points_list):
                        mp.add_observation(last_keyframe_idx, tuple(p_lkf_inliers[idx]))
                        mp.add_observation(i, tuple(p_ckf_inliers[idx]))

                    map_points.extend(new_map_points_list) #adds new map points to existing map points (map points grows quickly and the calculations get slower)
                    #map_points = new_map_points_list #replaces old map points with new ones

                    MAX_MAP_POINTS = 80
                    if len(map_points) > MAX_MAP_POINTS:
                        map_points = map_points[-MAX_MAP_POINTS:]

                    keyframe_history.append({
                        'id': i,
                        'rvec': rvec, # From PnP
                        'tvec': tvec  # From PnP
                    })
                    
                    # Keep history size constant
                    if len(keyframe_history) > MAX_HISTORY:
                        keyframe_history.pop(0)

                    # RUN BUNDLE ADJUSTMENT
                    if len(keyframe_history) >= MAX_HISTORY:
                        continue
                        print(f"Running Bundle Adjustment on window of {len(keyframe_history)} frames...")
                        hidden_state, observations, mapping = build_optimization_problem(map_points, keyframe_history)
                        if hidden_state is not None:
                            # 2. Run Optimization
                            # Note: You might need to verify if runBA fixes the first frame! 
                            # If not, the whole window might drift. 
                            optimized_state = runBA(hidden_state, observations, K)
                            
                            # 3. Update Real Objects
                            update_objects_from_state(optimized_state, map_points, keyframe_history, mapping)
                            print("Local BA complete.")


                #visualize_matches(p_lkf_inliers, p_ckf_inliers, last_keyframe, image, max_matches=30) #visualizes the matches between keyframes (debugging step)
                lkf_kp = key_points #saves current keypoints as last keyframe keypoints for next iteration
                lkf_dp = described_points #saves current descriptors as last keyframe descriptors for next iteration
                lkf_pm = projection_matrix #saves current projection matrix as last keyframe projection matrix for next iteration

                last_keyframe = image #saves current image as last keyframe for next iteration
                last_keyframe_idx = i
                print("Is a keyframe")
                #return map_points, new_map_points_list #(debugging step)
        
        elif is_keyframe and (i - start_index) == 0 and success: #checks if current frame is the first keyframe and PnP was successful
            lkf_kp = key_points #saves current keypoints as last keyframe keypoints for next iteration
            lkf_dp = described_points #saves current descriptors as last keyframe descriptors for next iteration
            lkf_pm = projection_matrix #saves current projection matrix as last keyframe projection matrix for next iteration
            last_keyframe = image #saves current image as last keyframe for next iteration
            print("First keyframe")

        else: #runs if current frame is not a keyframe
            print("Is not a keyframe")

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
    map_points, new_map_points_list = continuous_operation(ds, path_handle, last_frame, start_index, map_points)

""" #Visualization of old and new map points after last keyframe processing to compare (debugging purposes)
    final_viewer = WorldViewer2D(scale=3)
    map_points_3d_old = np.array([mp.position for mp in map_points]).T
    final_viewer.add_points(map_points_3d_old.T, color = 'blue')
    
    map_points = new_map_points_list
    map_points_3d_new = np.array([mp.position for mp in map_points]).T
    final_viewer.add_points(map_points_3d_new.T, color = 'red')
    final_viewer.draw()
    plt.ioff()
    plt.show()
"""
