import cv2
import numpy as np
from src.visualizations import visualize_world_points_2d, visualize_world_points_3d

class MapPoint():
    def __init__(self, position_3d, descriptor):
        self.position = position_3d
        self.descriptor = descriptor    # In our case SIFT
        self.observations = {}          # {frame_id : kp_idx} currently unsused, but maybe needed in bundle adjustment.


def calculate_essential_matrix_and_triangulate_map_points(p0, p1, descriptors, K, pos_camera_1):
    # Note from Markus: maybe we have to use the Fundamental Matrix and are not allowed to use the intrinsic Matrix K.
    # We need to clarify this.
    E, inliers = cv2.findEssentialMat(
        p0,
        p1,
        cameraMatrix=K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    number_of_matches = len(p0)
    number_of_inlier_points, R, t, mask = cv2.recoverPose(E, p0, p1, K)
    print(f"Number of matches: {number_of_matches}")
    print(f"Inliners found: {number_of_inlier_points}")
    print(f"Inlier ratio: {number_of_inlier_points / number_of_matches * 100 :.1f}%")

    # Remove the outlier points
    inlier_mask = mask.ravel().astype(bool)
    p0_inliers = p0[inlier_mask]
    p1_inliers = p1[inlier_mask]

    # Point triangulation:
    P1 = np.hstack((K, np.zeros((3, 1))))   # Projection Matrix of Camera 1
    P2 = np.hstack((K @ R, K @ t))          # Projection Matrix of Camera 2

    points_4d = cv2.triangulatePoints(P1, P2, p0_inliers.T, p1_inliers.T) # Points in homogenous coordinates
    points_3d = (points_4d[:3] / points_4d[3]).T  # transform to real 3D coordinates

    visualize_world_points_2d(points_3d, R, t)
    #visualize_world_points_3d(points_3d, R, t) # does not work properly for now..

    # Create MapPoints (landmarks)
    inlier_descriptors = descriptors[:, inlier_mask]
    map_points = [MapPoint(points_3d[i], inlier_descriptors[:,i]) for i in range(len(points_3d))]

    return map_points