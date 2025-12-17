import cv2
import numpy as np
import os
import glob

# Load calibration
K = np.load("K.npy")
dist = np.load("dist.npy")

# Input and output folders
input_folder = "frames"
output_folder = "undistorted"
os.makedirs(output_folder, exist_ok=True)

# Process all PNG frames
images = sorted(glob.glob(f"{input_folder}/*.png"))

for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1)
    print(newK)
    undistorted = cv2.undistort(img, K, dist, None, newK)

    # Crop to valid region
    x, y, w1, h1 = roi
    undistorted_cropped = undistorted[y:y+h1, x:x+w1]

    # Save
    outname = os.path.join(output_folder, os.path.basename(fname))
    cv2.imwrite(outname, undistorted_cropped)

print(f"Undistorted {len(images)} frames to '{output_folder}'")
