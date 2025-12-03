import cv2
import numpy as np


def visualize_matches_zoomed(p0, p1, img0, img1, zoom_radius=50, scale_factor=8, max_matches=10):
    
    if len(p0) == 0:
        return

    PATCH_SIZE = 2 * zoom_radius + 1 
    SCALED_SIZE = PATCH_SIZE * scale_factor
    img0_color = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    max_matches_to_show = min(max_matches, len(p0))

    CENTER_PT_SCALED = (zoom_radius * scale_factor, zoom_radius * scale_factor) 

    for i in range(max_matches_to_show):
        pt0 = (p0[i][0], p0[i][1]) 
        pt1 = (p1[i][0], p1[i][1]) 

        patch0 = cv2.getRectSubPix(img0_color, (PATCH_SIZE, PATCH_SIZE), pt0)
        patch1 = cv2.getRectSubPix(img1_color, (PATCH_SIZE, PATCH_SIZE), pt1)

        patch0_scaled = cv2.resize(patch0, (SCALED_SIZE, SCALED_SIZE), interpolation=cv2.INTER_NEAREST)
        patch1_scaled = cv2.resize(patch1, (SCALED_SIZE, SCALED_SIZE), interpolation=cv2.INTER_NEAREST) 
        cv2.circle(patch0_scaled, CENTER_PT_SCALED, 6, (0, 0, 255), -1) 
        cv2.circle(patch1_scaled, CENTER_PT_SCALED, 6, (255, 0, 0), -1) 
        
        combined_zoom_img = np.hstack((patch0_scaled, patch1_scaled))

        cv2.imshow(f"Match {i+1}/{max_matches_to_show} | Frame 0 (Red) vs Frame 1 (Blue) | {PATCH_SIZE}x{PATCH_SIZE} content scaled {scale_factor}x", combined_zoom_img)
        cv2.waitKey(0) 

    cv2.destroyAllWindows()

def visualize_matches(p0, p1, img0, img1, max_matches=10):

    print(f"Total Matches Found: {len(p0)}")
    
    if len(p0) == 0:
        return

    img0_col = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img1_col = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    h0, w0 = img0_col.shape[:2]
    h1, w1 = img1_col.shape[:2]

    canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0_col
    canvas[:h1, w0:w0+w1] = img1_col

    num_show = min(len(p0), max_matches)
    
    for i in range(num_show):
        pt0 = (int(p0[i, 0]), int(p0[i, 1]))
        
        pt1 = (int(p1[i, 0]) + w0, int(p1[i, 1]))

        cv2.circle(canvas, pt0, 4, (0, 0, 255), -1)
        cv2.circle(canvas, pt1, 4, (255, 0, 0), -1)

    cv2.imshow(f"Top {num_show} Matches", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_keypoints(key_points, img, window_name, dot_radius=3):
    """
    Draws all selected keypoints on the image.
    key_points: 2xN array of (row, col) coordinates.
    img: Grayscale image.
    """
    img_kp_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for i in range(key_points.shape[1]):
        pt_col = int(key_points[1, i])
        pt_row = int(key_points[0, i])
        
        cv2.circle(img_kp_vis, (pt_col, pt_row), dot_radius, (0, 255, 0), -1)

    print(f"Total keypoints selected for {window_name}: {key_points.shape[1]}")
    cv2.imshow(window_name, img_kp_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
