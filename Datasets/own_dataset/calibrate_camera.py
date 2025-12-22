import cv2
import numpy as np
import glob

CHECKERBOARD = (7, 9)     # inner corners (cols, rows)
SQUARE_SIZE = 0.019       # meters
IMAGE_PATH = "calibration_frames/*.png"

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points


images = glob.glob(IMAGE_PATH)

print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
              cv2.CALIB_CB_FAST_CHECK +
              cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # Refine corner locations
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-6
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpoints.append(corners)

        print(f"[OK] {fname}")
    else:
        print(f"[SKIP] {fname}")

# Calibration
print(len(objpoints), "valid calibration images found.")
assert len(objpoints) >= 10, "Not enough valid calibration images"

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# RESULTS
print("\n=== Calibration Results ===")
print("Reprojection error:", ret)
print("\nCamera matrix K:\n", K)
print("\nDistortion coefficients:\n", dist.ravel())

# SAVE RESULTS
np.save("K.npy", K)
np.save("dist.npy", dist)
