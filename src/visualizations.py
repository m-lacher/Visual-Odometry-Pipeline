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
    def __init__(
        self,
        map_size=700,
        zoom=20.0,
        map_radius=12.0,
        img_size=(480, 640),
        max_points_to_draw=500,
        draw_keypoints_enabled=True,
        plot_height=150
    ):
        """
        map_size : pixel size of the map window
        zoom     : pixels per meter
        map_radius : half-width of map (limits the area shown)
        plot_height : height of zoomed-out trajectory plot below image
        """
        self.map_size = map_size
        self.zoom = zoom
        self.map_radius = map_radius
        self.img_h, self.img_w = img_size
        self.max_points_to_draw = max_points_to_draw
        self.draw_keypoints_enabled = draw_keypoints_enabled
        self.plot_h = plot_height

        self.points = []               # list of (N,3)
        self.camera_positions = []     # list of (3,)
        self.current_image = None
        self.key_points = None

    def add_points(self, points_3d):
        self.points.append(np.asarray(points_3d))

    def clear_points(self):
        self.points = []

    def add_camera(self, t):
        self.camera_positions.append(t.flatten())

    def update_image(self, img, key_points=None):
        self.current_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.key_points = key_points

    def _world_to_map(self, x, z, cx, cz):
        u = int(self.map_size / 2 + (x - cx) * self.zoom)
        v = int(self.map_size / 2 - (z - cz) * self.zoom)
        return u, v

    def _inside_radius(self, x, z, cx, cz):
        return (
            abs(x - cx) <= self.map_radius and
            abs(z - cz) <= self.map_radius
        )

    def draw(self, window_name="VO Top-down Viewer"):
        canvas = np.zeros(
            (max(self.map_size, self.img_h + self.plot_h + 10), self.map_size + self.img_w, 3),
            dtype=np.uint8
        )

        # Map Panel
        map_img = canvas[:self.map_size, :self.map_size]
        map_img[:] = 25
        if self.camera_positions:
            cam_text = f"Top-down view (X-Z) | Camera pos (X,Z): ({self.camera_positions[-1][0]:.2f}, {self.camera_positions[-1][2]:.2f})"
        else:
            cam_text = "Top-down view (X-Z) | Camera pos: (0.00, 0.00)"

        cv2.putText(
            map_img,
            cam_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1
        )

        # Center on latest camera
        if self.camera_positions:
            cx, _, cz = self.camera_positions[-1]
        else:
            cx, cz = 0.0, 0.0

        # Draw map points
        if self.points:
            P = np.vstack(self.points)
            for x, _, z in P:
                if not self._inside_radius(x, z, cx, cz):
                    continue
                u, v = self._world_to_map(x, z, cx, cz)
                if 0 <= u < self.map_size and 0 <= v < self.map_size:
                    cv2.circle(map_img, (u, v), 2, (255, 0, 255), -1)

        # Draw camera trajectory
        if len(self.camera_positions) > 1:
            for i in range(1, len(self.camera_positions)):
                x0, _, z0 = self.camera_positions[i - 1]
                x1, _, z1 = self.camera_positions[i]

                if not (
                    self._inside_radius(x0, z0, cx, cz) or
                    self._inside_radius(x1, z1, cx, cz)
                ):
                    continue

                u0, v0 = self._world_to_map(x0, z0, cx, cz)
                u1, v1 = self._world_to_map(x1, z1, cx, cz)

                cv2.line(map_img, (u0, v0), (u1, v1), (0, 255, 0), 2)

        # Draw camera center
        uc, vc = self._world_to_map(cx, cz, cx, cz)
        cv2.circle(map_img, (uc, vc), 5, (0, 0, 255), -1)

        # Image Panel
        if self.current_image is not None:
            img_h_orig, img_w_orig = self.current_image.shape[:2]
            img_panel = cv2.resize(self.current_image, (self.img_w, self.img_h))

            if (
                self.key_points is not None
                and self.key_points.size > 0
                and self.draw_keypoints_enabled
            ):
                dp = self.key_points

                if dp.shape[0] != 2 and dp.shape[1] == 2:
                    dp = dp.T
                elif dp.shape[0] == 3:
                    dp = dp[:2] / dp[2]

                if dp.shape[1] > self.max_points_to_draw:
                    idx = np.linspace(
                        0, dp.shape[1] - 1,
                        self.max_points_to_draw,
                        dtype=int
                    )
                    dp = dp[:, idx]

                cols = np.clip(
                    (dp[1] * self.img_w / img_w_orig).astype(int),
                    0, self.img_w - 1
                )
                rows = np.clip(
                    (dp[0] * self.img_h / img_h_orig).astype(int),
                    0, self.img_h - 1
                )

                for r, c in zip(rows, cols):
                    cv2.drawMarker(
                        img_panel,
                        (c, r),
                        (0, 255, 0),
                        cv2.MARKER_CROSS,
                        4,
                        1
                    )

            canvas[:self.img_h, self.map_size:self.map_size + self.img_w] = img_panel


        # making the global trajectory graph
        plot_canvas = np.ones((self.plot_h, self.img_w, 3), dtype=np.uint8) * 25
        
        if len(self.camera_positions) > 1:
            cams = np.array(self.camera_positions)
            xs, zs = cams[:, 0], cams[:, 2]
            
            x_min, x_max = xs.min(), xs.max()
            z_min, z_max = zs.min(), zs.max()
            
            x_range = max(x_max - x_min, 0.1)
            z_range = max(z_max - z_min, 0.1)
            
            plot_margin = (self.img_w - 20) / x_range
            plot_margin_z = (self.plot_h - 20) / z_range
            dynamic_zoom = min(plot_margin, plot_margin_z)
            
            cx_plot = (x_min + x_max) / 2
            cz_plot = (z_min + z_max) / 2
            
            def world_to_plot(x, z):
                u = int(self.img_w / 2 + (x - cx_plot) * dynamic_zoom)
                v = int(self.plot_h / 2 - (z - cz_plot) * dynamic_zoom)
                return u, v
            
            for i in range(1, len(cams)):
                u0, v0 = world_to_plot(cams[i-1, 0], cams[i-1, 2])
                u1, v1 = world_to_plot(cams[i, 0], cams[i, 2])
                if 0 <= u0 < self.img_w and 0 <= v0 < self.plot_h and 0 <= u1 < self.img_w and 0 <= v1 < self.plot_h:
                    cv2.line(plot_canvas, (u0, v0), (u1, v1), (0, 255, 255), 1)
            
            u_curr, v_curr = world_to_plot(cams[-1, 0], cams[-1, 2])
            cv2.circle(plot_canvas, (u_curr, v_curr), 3, (0, 0, 255), -1)
        
        cv2.putText(plot_canvas, "Full Trajectory (X-Z)", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        canvas[self.img_h + 10:self.img_h + 10 + self.plot_h, self.map_size:self.map_size + self.img_w] = plot_canvas

        cv2.imshow(window_name, canvas)
        cv2.waitKey(1)