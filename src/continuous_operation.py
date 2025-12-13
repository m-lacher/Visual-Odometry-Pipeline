import cv2
import os
import numpy as np
from src.initialization import process_frame, match_points
from src.visualizations import WorldViewer2D
from src.helpers.map_points import MapPoint, calculate_essential_matrix_and_triangulate_map_points
from src.helpers.match_descriptors import matchDescriptorsLOWE


def get_image_path(ds, path_handle, frame_index):
    """Get the image path for a given frame based on dataset type."""
    if ds == 0:
        return os.path.join(path_handle, '05', 'image_0', f"{frame_index:06d}.png")
    elif ds == 1:
        return path_handle[frame_index]
    elif ds == 2:
        return os.path.join(path_handle, 'images', f"img_{frame_index:05d}.png")
    elif ds == 3:
        return os.path.join(path_handle, f"{frame_index:06d}.png")
    else:
        raise ValueError("Invalid dataset index")


def continuous_operation(ds, path_handle, last_frame, start_index, map_points, K):
    """
    Continuously process frames, match features, estimate pose, and triangulate new landmarks.
    
    Args:
        ds: Dataset index (0: KITTI, 1: Malaga, 2: Parking, 3: Own)
        path_handle: Path or list of paths to dataset
        last_frame: Last frame index to process
        start_index: Starting frame index
        map_points: Initial map points from initialization
        K: Camera intrinsic matrix
    
    Returns:
        map_points: Updated list of map points
    """
    import os
    
    keyframe_dist = 4  # defines every nth Frame is a keyframe
    keyframe_history = []
    MAX_HISTORY = 6  # bundle window
    last_keyframe_idx = start_index

    viewer = WorldViewer2D()  # for visualization
    map_points_3d = np.array([mp.position for mp in map_points]).T
    viewer.add_points(map_points_3d.T)
    MAX_MAP_POINTS = 80
    
    # Initialize tracking variables
    lkf_kp = None
    lkf_dp = None
    lkf_pm = None
    prev_matched_dp = None
    prev_matched_kp3d = None
    
    for i in range(start_index, last_frame + 1):
        print(f"\nProcessing frame {i}")
        print("=====================")
        
        image_path = get_image_path(ds, path_handle, i)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        is_keyframe = (i - start_index) % keyframe_dist == 0
        
        if image is None:
            print(f"Warning: could not read {image_path}")
            continue

        
        height, width = image.shape[:2]  # img.shape = (H, W) for grayscale, (H, W, C) for color
        print(f"Image dimensions: width={width}, height={height}")

        # display the mapped points
        map_points_3d = np.array([mp.position for mp in map_points]).T
        viewer.add_points(map_points_3d.T)

        # describe new image features
        key_points, described_points = process_frame(img=image)

        viewer.update_image(image, key_points)
        
        # find new matches from existing landmarks
        map_descriptors = np.array([mp.descriptor for mp in map_points]).T

        matches = matchDescriptorsLOWE(described_points, map_descriptors, match_lambda=0.7)
        query_indices = np.nonzero(matches >= 0)[0]
        match_indices = matches[query_indices].astype(int)
        points_matched_3d = map_points_3d[:, match_indices]
        points_matched_2d = key_points[:, query_indices]
        points_matched_2d = points_matched_2d[::-1, :].T  # shape (N,2) for PnP
        points_matched_3d = points_matched_3d.T  # shape (N,3) for PnP
        
        # do PnP with Ransac
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.array(points_matched_3d),
            np.array(points_matched_2d),
            K,
            None
        )
        print(f"PnP Success: {success}")
        
        # If pnp was successful: Plot new camera pose
        if success:
            print(f"Number of inliers found: {len(inliers)}")
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)

            R_wc = R.T
            t_wc = -R_wc @ t
            projection_matrix = np.hstack((K @ R, K @ t))

            viewer.add_camera(t_wc) # only position
            viewer.draw()

            for idx in inliers.ravel():
                map_idx = match_indices[idx]
                mp = map_points[map_idx]
                uv = tuple(points_matched_2d[idx])
                mp.add_observation(i, uv)
        
        # Keyframe processing
        if is_keyframe and (i - start_index) != 0 and success:
            viewer.clear_points()
            p_lkf, p_ckf, p_lkf_descriptors, p_ckf_descriptors = match_points(
                lkf_kp, lkf_dp, key_points, described_points, match_lambda=0.7
            )
            
            if len(p_lkf) < 8:
                print("Triangulation failed")
            else:
                F, mask = cv2.findEssentialMat(
                    p_lkf, p_ckf, cameraMatrix=K, method=cv2.RANSAC, 
                    prob=0.999, threshold=1.0
                )
                
                if mask is None:
                    print("triangulation failed")
                else:
                    inlier_mask = mask.ravel().astype(bool)
                    p_lkf_inliers = p_lkf[inlier_mask]
                    p_ckf_inliers = p_ckf[inlier_mask]
                    inlier_descriptors = p_ckf_descriptors[:, inlier_mask]

                    new_map_points = cv2.triangulatePoints(
                        lkf_pm, projection_matrix, p_lkf_inliers.T, p_ckf_inliers.T
                    )
                    new_map_points_normalized = (new_map_points[:3] / new_map_points[3]).T
                    new_map_points_normalized_CF = R @ new_map_points_normalized.T + t
                    
                    valid_point_mask_min = new_map_points_normalized_CF[2, :] > 0
                    valid_point_mask_max = new_map_points_normalized_CF[2, :] < 30
                    valid_point_mask = valid_point_mask_min & valid_point_mask_max

                    new_map_points_normalized = new_map_points_normalized[valid_point_mask]
                    inlier_descriptors = inlier_descriptors[:, valid_point_mask]
                    new_map_points_list = [
                        MapPoint(new_map_points_normalized[s], inlier_descriptors[:, s])
                        for s in range(len(new_map_points_normalized))
                    ]

                    for idx, mp in enumerate(new_map_points_list):
                        mp.add_observation(last_keyframe_idx, tuple(p_lkf_inliers[idx]))
                        mp.add_observation(i, tuple(p_ckf_inliers[idx]))

                    map_points.extend(new_map_points_list)
                    if len(map_points) > MAX_MAP_POINTS:
                        map_points = map_points[-MAX_MAP_POINTS:]

                    keyframe_history.append({
                        'id': i,
                        'rvec': rvec,
                        'tvec': tvec
                    })
                    
                    if len(keyframe_history) > MAX_HISTORY:
                        keyframe_history.pop(0)

                    lkf_kp = key_points
                    lkf_dp = described_points
                    lkf_pm = projection_matrix
                    last_keyframe_idx = i
                    print("Is a keyframe")
        
        elif is_keyframe and (i - start_index) == 0 and success:
            lkf_kp = key_points
            lkf_dp = described_points
            lkf_pm = projection_matrix
            print("First keyframe")
        else:
            print("Is not a keyframe")

        cv2.waitKey(10)

    return map_points