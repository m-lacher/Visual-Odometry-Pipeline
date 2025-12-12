

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# --- Helper Functions for Geometry ---

def twist_to_homog(twist):
    """Convert 6D twist (v, w) to 4x4 Homogeneous Matrix."""
    v, w = twist[:3], twist[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        R = np.eye(3)
        V = np.eye(3)
    else:
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        R = np.eye(3) + (np.sin(theta)/theta)*K + ((1-np.cos(theta))/theta**2)*(K@K)
        V = np.eye(3) + ((1-np.cos(theta))/theta**2)*K + ((theta-np.sin(theta))/theta**3)*(K@K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T

def homog_to_twist(T):
    """Convert 4x4 Homogeneous Matrix to 6D twist (v, w)."""
    R = T[:3, :3]
    t = T[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if theta < 1e-10:
        w = np.zeros(3)
        v = t
    else:
        w = theta / (2 * np.sin(theta)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        V_inv = np.eye(3) - 0.5*K + (1 - theta/(2*np.tan(theta/2)))/(theta**2)*(K@K)
        v = V_inv @ t
    return np.concatenate((v, w))

def project_points(points_3d, twist, K):
    """Project 3D world points to 2D using a camera pose (twist)."""
    T_wc = twist_to_homog(twist) # Pose of camera in world
    T_cw = np.linalg.inv(T_wc)   # World to Camera
    
    # Transform points: X_c = R*X_w + t
    points_c = (T_cw[:3, :3] @ points_3d.T).T + T_cw[:3, 3]
    
    # Project: u = fx*(x/z) + cx
    z = points_c[:, 2]
    # Avoid division by zero
    z[np.abs(z) < 1e-6] = 1e-6
    
    u = K[0,0] * (points_c[:, 0] / z) + K[0,2]
    v = K[1,1] * (points_c[:, 1] / z) + K[1,2]
    
    return np.column_stack((u, v))

# --- Main Bundle Adjustment Function ---

def bundle_adjustment(map_points, keyframe_history, K):
    """
    Robust Bundle Adjustment using native OpenCV rvec/tvec parameterization.
    """
    # 0. Safety Check
    if len(keyframe_history) < 2:
        return map_points, keyframe_history

    # Ensure K is float64 to avoid OpenCV type errors
    K = K.astype(np.float64)

    # 1. Select Data
    # --------------
    kf_map = {kf['id']: idx for idx, kf in enumerate(keyframe_history)}
    
    active_mp_indices = [] 
    observations = []      # (window_cam_idx, active_pt_idx, u, v)

    for i, mp in enumerate(map_points):
        valid_obs = []
        for fid, uv in mp.observations.items():
            if fid in kf_map:
                valid_obs.append((kf_map[fid], uv))
        
        # Only optimize points seen by at least 2 cameras in the window
        if len(valid_obs) >= 2:
            active_pt_idx = len(active_mp_indices)
            active_mp_indices.append(i)
            for cam_idx, uv in valid_obs:
                observations.append((cam_idx, active_pt_idx, uv[0], uv[1]))

    observations = np.array(observations)
    n_points = len(active_mp_indices)
    n_cameras = len(keyframe_history)

    if n_points == 0:
        return map_points, keyframe_history

    # 2. Build State Vector
    # ---------------------
    # We optimize: [rvec_1, tvec_1, rvec_2, tvec_2, ..., point_0, point_1, ...]
    # Camera 0 is FIXED (anchor) and not included in 'x0'.
    
    x0_cameras = []
    # Skip index 0 (Fixed Camera)
    for i in range(1, n_cameras): 
        kf = keyframe_history[i]
        r = kf['rvec'].flatten()
        t = kf['tvec'].flatten()
        x0_cameras.extend(list(r) + list(t)) # 6 params per camera

    # Add Points (3 params per point)
    x0_points = np.array([map_points[i].position for i in active_mp_indices]).flatten()
    
    x0 = np.concatenate((x0_cameras, x0_points))

    # 3. Optimization Setup
    # ---------------------
    n_opt_cameras = n_cameras - 1
    n_cam_params = n_opt_cameras * 6
    
    # Pre-cache Camera 0 (Fixed)
    fixed_rvec = keyframe_history[0]['rvec'].flatten()
    fixed_tvec = keyframe_history[0]['tvec'].flatten()

    # Residual Function
    def fun(x):
        # 1. Unpack Cameras
        # Reshape to (N, 6) -> [rvec, tvec]
        cameras_opt = x[:n_cam_params].reshape(n_opt_cameras, 6)
        
        # 2. Unpack Points
        points_3d = x[n_cam_params:].reshape(n_points, 3)
        
        residuals = []
        
        # We process observations one by one (or batched per camera for speed)
        # For clarity and safety against indexing errors, we loop observations:
        
        # Pre-transform points? No, OpenCV projectPoints handles rvec/tvec directly.
        # But calling projectPoints inside a loop is slow. 
        # Optimization: Project all points for each camera.
        
        projected_cache = {} # Map (cam_idx) -> projected_points_array
        
        # Reconstruct ALL camera poses (Fixed + Optimized)
        all_rvecs = [fixed_rvec]
        all_tvecs = [fixed_tvec]
        
        for i in range(n_opt_cameras):
            all_rvecs.append(cameras_opt[i, :3])
            all_tvecs.append(cameras_opt[i, 3:])
            
        # Compute residuals
        # Note: We can't easily vectorise across different cameras, 
        # so we iterate through observations.
        
        # To make this fast with OpenCV, we project the specific 3D point for that observation
        for i in range(len(observations)):
            cam_idx = int(observations[i, 0])
            pt_idx = int(observations[i, 1])
            obs_uv = observations[i, 2:]
            
            # Get Pose
            rvec = all_rvecs[cam_idx]
            tvec = all_tvecs[cam_idx]
            
            # Get Point
            Xw = points_3d[pt_idx]
            
            # Project (This is safe and robust)
            # projectPoints expects shape (N, 1, 3) or (N, 3)
            # It returns (N, 1, 2)
            pt_proj, _ = cv2.projectPoints(Xw.reshape(1, 1, 3), rvec, tvec, K, None)
            pt_proj = pt_proj.flatten() # (u, v)
            
            residuals.append(pt_proj - obs_uv)
            
        return np.concatenate(residuals)

    # Sparsity Matrix (Jacobian structure)
    m = 2 * len(observations)
    n = len(x0)
    A = lil_matrix((m, n), dtype=int)
    
    obs_cam_indices = observations[:, 0].astype(int)
    obs_pt_indices = observations[:, 1].astype(int)
    
    for i in range(len(observations)):
        c_idx = obs_cam_indices[i]
        p_idx = obs_pt_indices[i]
        
        # Row indices in residual vector
        r_idx = 2 * i
        
        # Camera derivatives (if not fixed cam 0)
        if c_idx > 0:
            opt_c_idx = c_idx - 1
            c_col_start = opt_c_idx * 6
            A[r_idx:r_idx+2, c_col_start : c_col_start+6] = 1
            
        # Point derivatives
        p_col_start = n_cam_params + (p_idx * 3)
        A[r_idx:r_idx+2, p_col_start : p_col_start+3] = 1

    # 4. Run Optimization
    # -------------------
    # ftol=1e-2 is usually enough for visual SLAM and prevents over-optimizing noise
    try:
        res = least_squares(fun, x0, jac_sparsity=A, verbose=0, x_scale='jac', ftol=1e-2, method='trf', loss='huber', f_scale=1.0)
    except Exception as e:
        print(f"Bundle Adjustment Failed: {e}")
        return map_points, keyframe_history

    # 5. Update Results
    # -----------------
    x_opt = res.x
    cameras_opt = x_opt[:n_cam_params].reshape(n_opt_cameras, 6)
    points_opt = x_opt[n_cam_params:].reshape(n_points, 3)
    
    # Update Points (IN PLACE)
    for i, idx_in_map in enumerate(active_mp_indices):
        map_points[idx_in_map].position = points_opt[i]
        
    # Update Keyframes
    for i in range(n_opt_cameras):
        # Remember: i=0 in opt_cameras corresponds to camera index 1 in history
        kf_idx = i + 1 
        keyframe_history[kf_idx]['rvec'] = cameras_opt[i, :3].reshape(3, 1)
        keyframe_history[kf_idx]['tvec'] = cameras_opt[i, 3:].reshape(3, 1)

    print(f"BA: Optimized {n_cameras} frames and {n_points} points. Error: {res.cost:.2f}")
    
    return map_points, keyframe_history