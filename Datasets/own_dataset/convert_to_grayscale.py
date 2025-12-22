import cv2
import glob
import os

input_folder = "undistorted_grayscale"
output_folder = "undistorted_grayscale"
os.makedirs(output_folder, exist_ok=True)

images = sorted(glob.glob(f"{input_folder}/*.png"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outname = os.path.join(output_folder, os.path.basename(fname))
    cv2.imwrite(outname, gray)

print(f"Converted {len(images)} images to grayscale")