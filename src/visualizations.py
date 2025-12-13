import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.helpers.draw_camera import drawCamera


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
    #cv2.destroyAllWindows()

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

def visualize_world_points_3d(points_3d, R, t, scale=10):
    """
    Visualize 3D points and camera poses.
    
    points_3d: Nx3 array of 3D points
    R, t: rotation and translation from recoverPose
    scale: length of camera arrows
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot 3D points
    points_3d = np.asarray(points_3d)
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], s=5, color='m')

    # --------------------------
    # Draw first camera at origin
    # --------------------------
    drawCamera(ax, np.zeros(3), np.eye(3), length_scale=scale, head_size=10)

    # --------------------------
    # Draw second camera
    # --------------------------
    t = np.asarray(t).reshape(3,)
    pos = (-R.T @ t).reshape(3,)
    drawCamera(ax, pos, R.T, length_scale=scale, head_size=10)

    # --------------------------
    # Label axes
    # --------------------------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Poses")
    ax.set_box_aspect([1,1,1])  # equal aspect ratio

    plt.show()

def visualize_world_points_2d(points_3d, R, t, scale=5):
    """
    2D visualization of points (X,Z) and camera positions (top-down view). (X is to the right, Z is forward from camera perspective)

    points_3d: Nx3 array of 3D points
    R, t: rotation and translation of the second camera
    scale: optional, for plotting camera axes as lines
    """
    points_3d = np.asarray(points_3d)
    
    fig, ax = plt.subplots()
    
    # Plot points
    ax.scatter(points_3d[:,0], points_3d[:,2], s=5, color='m', label='3D Points')

    # Camera 1 at origin
    cam1_pos = np.zeros(3)
    ax.scatter(cam1_pos[0], cam1_pos[1], color='r', s=50, label='Camera 1')
    # Optionally show orientation in 2D
    ax.arrow(cam1_pos[0], cam1_pos[1], scale, 0, color='r', head_width=0.05*scale)
    ax.arrow(cam1_pos[0], cam1_pos[1], 0, scale, color='g', head_width=0.05*scale)

    # Camera 2
    t = np.asarray(t).reshape(3,)
    cam2_pos = (-R.T @ t).reshape(3,)
    ax.scatter(cam2_pos[0], cam2_pos[1], color='b', s=50, label='Camera 2')
    # Draw axes in 2D (top-down)
    R2 = R.T
    ax.arrow(cam2_pos[0], cam2_pos[1], R2[0,0]*scale, R2[1,0]*scale, color='r', head_width=0.05*scale)
    ax.arrow(cam2_pos[0], cam2_pos[1], R2[0,1]*scale, R2[1,1]*scale, color='g', head_width=0.05*scale)
    
    # Labels & legend
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Top-down 2D view of points and cameras')
    ax.legend()
    ax.axis('equal')
    plt.show()


class WorldViewer2D:
    def __init__(self, scale=1.0):
        self.scale = scale

        # 3 subplots: top-down (XZ), front (XY), image
        self.fig, (self.ax_xz, self.ax_xy, self.ax_img) = plt.subplots(1, 3, figsize=(16, 5))

        # store points and cameras
        self.all_points = []
        self.camera_positions = []
        self.camera_rotations = []

        # XZ view
        self.ax_xz.set_xlabel("X")
        self.ax_xz.set_ylabel("Z")
        self.ax_xz.set_title("Top-down view (X-Z)")
        self.ax_xz.axis("equal")

        # XY view
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.set_title("Front view (X-Y)")
        self.ax_xy.axis("equal")

        # image view
        self.ax_img.axis("off")
        self.ax_img.set_title("Current Frame")

        plt.ion()
        plt.show()

    def add_points(self, points_3d):
        points_3d = np.asarray(points_3d)
        self.all_points.append(points_3d)

    def add_camera(self, R, t):
        cam_pos = t.flatten()
        self.camera_positions.append(cam_pos)
        self.camera_rotations.append(np.asarray(R))

    def update_image(self, img):
        self.ax_img.cla()
        self.ax_img.imshow(img, cmap="gray")
        self.ax_img.set_title("Current Frame")
        self.ax_img.axis("off")

    def clear_points(self):
        """Clear all stored 3D points."""
        self.all_points = []

    def draw(self):
        """Draw XZ map + XY map + image."""
        # --- CLEAR MAP AXES (but keep image) ---
        self.ax_xz.cla()
        self.ax_xy.cla()

        # XZ labels
        self.ax_xz.set_xlabel("X")
        self.ax_xz.set_ylabel("Z")
        self.ax_xz.set_title("Top-down view (X-Z)")
        self.ax_xz.axis("equal")

        # XY labels
        self.ax_xy.set_xlabel("X")
        self.ax_xy.set_ylabel("Y")
        self.ax_xy.set_title("Front view (X-Y)")
        self.ax_xy.axis("equal")

        # --- DRAW POINTS ---
        if len(self.all_points) > 0:
            P = np.vstack(self.all_points)
            self.ax_xz.scatter(P[:, 0], P[:, 2], s=5, color='m')
            self.ax_xy.scatter(P[:, 0], P[:, 1], s=5, color='m')

        # --- DRAW CAMERAS ---
        for i, cam_pos in enumerate(self.camera_positions):
            x, y, z = cam_pos
            R_cw = self.camera_rotations[i]

            # XZ view
            self.ax_xz.scatter(x, z, s=40, color='b')
            self.ax_xz.text(x, z, f"{i}", fontsize=8)

            cam_x_axis = R_cw[:, 0] * self.scale
            cam_z_axis = R_cw[:, 2] * self.scale

            self.ax_xz.arrow(x, z, cam_x_axis[0], cam_x_axis[2], color='r', head_width=0.05*self.scale)
            self.ax_xz.arrow(x, z, cam_z_axis[0], cam_z_axis[2], color='g', head_width=0.05*self.scale)

            # XY view
            self.ax_xy.scatter(x, y, s=40, color='b')
            self.ax_xy.text(x, y, f"{i}", fontsize=8)

            cam_y_axis = R_cw[:, 1] * self.scale

            self.ax_xy.arrow(x, y, cam_x_axis[0], cam_x_axis[1], color='r', head_width=0.05*self.scale)
            self.ax_xy.arrow(x, y, cam_y_axis[0], cam_y_axis[1], color='g', head_width=0.05*self.scale)

        plt.pause(0.001)