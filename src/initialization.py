from src.helpers.harris import harris
from src.helpers.select_keypoints import selectKeypoints
from src.helpers.describe_keypoints import describeKeypointsSIFT
from src.helpers.match_descriptors import matchDescriptorsLOWE
import numpy as np
import cv2

def process_frame(img, 
        num_keypoints=1000, # Note from Markus: I increased it to have more matches in initialization. 500 for sift, 1500 for orb
        detector='gftt',
        descriptor='sift' #when switching descriptor check MAX_MAP_POINTS and keyframe_dist in continuous_operation, num_keypoints above, and play with filter values in continuous_operation
    ):
    
    if detector == 'harris':
        harris_score = harris(img, patch_size=9, kappa=0.08)
        key_points = selectKeypoints(harris_score, num_keypoints, r=8)

    elif detector == 'gftt':
        corners = cv2.goodFeaturesToTrack(img, maxCorners=num_keypoints, qualityLevel=0.01, minDistance=10)
        key_points = corners.reshape(-1,2).T[::-1, :]

    elif detector == 'fast':
        fast = cv2.FastFeatureDetector_create(threshold=20)
        kp = fast.detect(img, None)
        kp = sorted(kp, key=lambda x: x.response, reverse=True)[:num_keypoints]
        key_points = np.array([[p.pt[1], p.pt[0]] for p in kp]).T

    elif detector == 'orb':
        orb = cv2.ORB_create(nfeatures=num_keypoints)
        kp = orb.detect(img, None)
        key_points = np.array([[p.pt[1], p.pt[0]] for p in kp]).T

    if descriptor == 'sift':
        described_points = describeKeypointsSIFT(img, key_points, 9)

    elif descriptor == 'orb':
        orb = cv2.ORB_create()
        cv_kp = [cv2.KeyPoint(p[1], p[0], 7) for p in key_points.T]
        _, descriptors = orb.compute(img, cv_kp)
        described_points = descriptors.T
    
    return key_points, described_points, descriptor

def match_points(key_points_0, described_points_0, key_points_1, described_points_1, descriptor_type, match_lambda):
    # WE MATCH POINTS FROM IMG 1 TO POINTS FROM IMG 0
    # lambda of 0.7 is pretty good for now. lower -> more strict matches
    matches = matchDescriptorsLOWE(described_points_1, described_points_0, descriptor_type, match_lambda=0.7)
    
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices].astype(int)

    p0_coords = key_points_0[:, match_indices] 
    p1_coords = key_points_1[:, query_indices]

    p0_descriptor_matched = described_points_0[:, match_indices]
    p1_descriptor_matched = described_points_1[:, query_indices]

    p0 = p0_coords[::-1, :].T 
    p1 = p1_coords[::-1, :].T
    
    return p0, p1, p0_descriptor_matched, p1_descriptor_matched