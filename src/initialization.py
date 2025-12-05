from src.helpers.harris import harris
from src.helpers.select_keypoints import selectKeypoints
from src.helpers.describe_keypoints import describeKeypointsSIFT
from src.helpers.match_descriptors import matchDescriptorsLOWE
import numpy as np

def process_frame(img, 
        corner_patch_size=9, 
        harris_kappa=0.08, 
        num_keypoints=200, 
        nonmaximum_supression_radius=8, 
        descriptor_radius=9
    ):
    
    harris_score = harris(img, corner_patch_size, harris_kappa)
    
    key_points = selectKeypoints(harris_score, num_keypoints, nonmaximum_supression_radius)
    
    described_points = describeKeypointsSIFT(img, key_points, descriptor_radius)
    
    
    return key_points, described_points

def match_points(key_points_0, described_points_0, key_points_1, described_points_1, match_lambda):
    # WE MATCH POINTS FROM IMG 1 TO POINTS FROM IMG 0
    # lambda of 0.7 is pretty good for now. lower -> more strict matches
    matches = matchDescriptorsLOWE(described_points_1, described_points_0, match_lambda=0.7)
    
    query_indices = np.nonzero(matches >= 0)[0]
    match_indices = matches[query_indices].astype(int)

    p0_coords = key_points_0[:, match_indices] 
    p1_coords = key_points_1[:, query_indices]

    p0_descriptor_matched = described_points_0[:, match_indices]
    p1_descriptor_matched = described_points_1[:, match_indices]

    p0 = p0_coords[::-1, :].T 
    p1 = p1_coords[::-1, :].T
    
    return p0, p1, p0_descriptor_matched, p1_descriptor_matched