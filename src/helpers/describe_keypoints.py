import numpy as np
import cv2

def describeKeypoints(img, keypoints, r):
    # This is the descriptor used in the solution of E3 (it is horrible and not invariant to anything making it almost useless)
    pass
    N = keypoints.shape[1]
    desciptors = np.zeros([(2*r+1)**2, N])
    padded = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(N):
        kp = keypoints[:, i].astype(int) + r
        desciptors[:, i] = padded[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)].flatten()
        
    return desciptors

def describeKeypointsSIFT(img, keypoints, r=None):
    # this function takes raw coordinates of the keypoints and the image. r is not used in sift
    # we use openCV SIFT to describe the keypoints. This makes them invariant to most things

    # we generate keypoint_objects which are required by the opencv implementation of sift
    n = keypoints.shape[1]
    print("keypoints:", n)
    keypoint_objects = []
    for i in range(n):
        x = float(keypoints[1, i])
        y = float(keypoints[0, i])
        keypoint_objects.append(cv2.KeyPoint(x, y, size=21.0))
        
    sift_object = cv2.SIFT_create()
    keypoints_out, descriptors_out = sift_object.compute(img, keypoint_objects)
    
    return descriptors_out.T # This needs to be transposed because sift outputs (Nx128) and we want (128xN) as done in the exercises