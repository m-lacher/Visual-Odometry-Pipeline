import os
import numpy as np
from glob import glob

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
        left_images = sorted(glob(os.path.join(img_dir, '*left.jpg')))
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