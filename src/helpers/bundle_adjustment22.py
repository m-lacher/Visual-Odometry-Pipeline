import numpy as np
import cv2
import scipy.sparse
from scipy.optimize import least_squares
from src.helpers.utils import twist2HomogMatrix, HomogMatrix2twist, projectPoints

def run_bundle_adjustment(map_points, keyframe_history, K, fix_first_frame=True):
    """
    Runs Local Bundle Adjustment on the active window of keyframes and map points.
    
    map_points: List of all MapPoint objects.
    keyframe_history: List of dicts [{'id': int, 'rvec': np.array, 'tvec': np.array}, ...]
    K: Intrinsic matrix (3x3).
    fix_first_frame: If True, the oldest keyframe in the window is not optimized (Anchor).
    """
    
    # 1. Setup: Identification & Data Filtering
    # -----------------------------------------
    # Map frame_id to the index in our optimization window (0 to N-1)
    kf_id_to_idx = {kf['id']: idx for idx, kf in enumerate(keyframe_history)}
    n_keyframes = len(keyframe_history)
    
    if n_keyframes < 2:
        return # Not enough frames to optimize

    # Filter MapPoints: Keep only those observed by at least 2 keyframes in the window
    active_map_points = []
    observations = [] # List of tuples: (cam_idx, point_idx, u, v)

    for mp_idx, mp in enumerate(map_points):
        # Find which of the active keyframes observe this point
        valid_obs = []
        for fid, uv in mp.observations.items():
            if fid in kf_id_to_idx:
                cam_idx = kf_id_to_idx[fid]
                valid_obs.append((cam_idx, uv))
        
        if len(valid_obs) >= 2:
            # This point is valid for optimization
            current_point_idx = len(active_map_points)
            active_map_points.append(mp)
            
            for cam_idx, uv in valid_obs:
                observations.append((cam_idx, current_point_idx, uv[0], uv[1]))

    n_points = len(active_map_points)
    if n_points == 0:
        return

    observations = np.array(observations) # Format: [cam_idx, pt_idx, u, v]

    # 2. State Vector Construction
    # ----------------------------
    # We need to convert R,t (Camera->World) to Twists of T_WC (World->Camera Pose)
    # The 'runBA.py' logic optimizes T_WC (Pose), so we project X_w to X_c using inv(T_WC).
    
    twist_list = []
    fixed_pose_T_WC = None # Store the fixed pose if needed

    for i, kf in enumerate(keyframe_history):
        # Current: rvec, tvec are T_CW (World to Camera)
        R_cw, _ = cv2.Rodrigues(kf['rvec'])
        t_cw = kf['tvec'].reshape(3, 1)
        
        T_CW = np.eye(4)
        T_CW[:3, :3] = R_cw
        T_CW[:3, 3] = t_cw[:, 0]
        
        # Optimization uses T_WC (Pose)
        T_WC = np.linalg.inv(T_CW)
        
        if fix_first_frame and i == 0:
            fixed_pose_T_WC = T_WC
            continue # Don't add to state vector
            
        twist = HomogMatrix2twist(T_WC)
        twist_list.append(twist)

    # Initial State: [Twists..., Points...]
    x0_twists = np.concatenate(twist_list) if len(twist_list) > 0 else np.array([])
    x0_points = np.array([mp.position for mp in active_map_points]).flatten()
    
    x0 = np.concatenate((x0_twists, x0_points))

    # Number of cameras actually being optimized
    n_opt_cameras = n_keyframes - 1 if fix_first_frame else n_keyframes

    # 3. Sparsity Matrix
    # ------------------
    # Jacobian size: (2 * n_observations) x (6 * n_opt_cameras + 3 * n_points)
    m = 2 * len(observations)
    n = 6 * n_opt_cameras + 3 * n_points
    pattern = scipy.sparse.lil_matrix((m, n), dtype=int)

    obs_idx = 0
    for i in range(len(observations)):
        cam_idx = int(observations[i, 0])
        pt_idx = int(observations[i, 1])
        
        # Handle index shift if first camera is fixed
        opt_cam_idx = cam_idx - 1 if fix_first_frame else cam_idx
        
        # If this observation belongs to the fixed camera, it only affects the point parameters
        if fix_first_frame and cam_idx == 0:
            # Depends only on point
            pattern[2*i : 2*i+2, 6*n_opt_cameras + 3*pt_idx : 6*n_opt_cameras + 3*(pt_idx+1)] = 1
        else:
            # Depends on camera (6 params) and point (3 params)
            if opt_cam_idx >= 0:
                pattern[2*i : 2*i+2, 6*opt_cam_idx : 6*(opt_cam_idx+1)] = 1
            pattern[2*i : 2*i+2, 6*n_opt_cameras + 3*pt_idx : 6*n_opt_cameras + 3*(pt_idx+1)] = 1
            
    # 4. Error Function
    # -----------------
    def fun(x):
        # Unpack state
        # 1. Reconstruct Poses (T_WC)
        current_twists = x[:6*n_opt_cameras].reshape((n_opt_cameras, 6))
        
        poses_T_WC = []
        twist_idx = 0
        
        for i in range(n_keyframes):
            if fix_first_frame and i == 0:
                poses_T_WC.append(fixed_pose_T_WC)
            else:
                poses_T_WC.append(twist2HomogMatrix(current_twists[twist_idx]))
                twist_idx += 1
                
        # 2. Reconstruct Points
        current_points = x[6*n_opt_cameras:].reshape((n_points, 3))
        
        residuals = []
        
        # Compute errors
        # To vectorise: process per camera
        for i in range(len(observations)):
            cam_idx = int(observations[i, 0])
            pt_idx = int(observations[i, 1])
            uv_obs = observations[i, 2:] # (u, v)
            
            T_WC = poses_T_WC[cam_idx]
            P_W = current_points[pt_idx]
            
            # Project: T_CW * P_W
            T_CW = np.linalg.inv(T_WC)
            P_C = T_CW[:3, :3] @ P_W + T_CW[:3, 3]
            
            proj = projectPoints(P_C.reshape(3, 1), K).flatten()
            
            residuals.append(proj - uv_obs)
            
        return np.concatenate(residuals)

    # 5. Optimization
    # ---------------
    print(f"BA: Optimizing {n_opt_cameras} frames and {n_points} points...")
    #res = least_squares(fun, x0, jac_sparsity=pattern, verbose=0, x_scale='jac', ftol=1e-3, method='trf')
    res = least_squares(
        fun, 
        x0, 
        jac_sparsity=pattern, 
        verbose=0, 
        x_scale='jac', 
        ftol=1e-3, 
        method='trf', 
        loss='huber',
        f_scale=1.0
    )
    # 6. Update Objects
    # -----------------
    x_opt = res.x
    opt_twists = x_opt[:6*n_opt_cameras].reshape((n_opt_cameras, 6))
    opt_points = x_opt[6*n_opt_cameras:].reshape((n_points, 3))
    
    # Update Points
    for i, mp in enumerate(active_map_points):  
        mp.position = opt_points[i]
        
    # Update Keyframes
    twist_idx = 0
    for i, kf in enumerate(keyframe_history):
        if fix_first_frame and i == 0:
            continue
            
        T_WC_opt = twist2HomogMatrix(opt_twists[twist_idx])
        twist_idx += 1
        
        # Convert back to T_CW for storage
        T_CW_opt = np.linalg.inv(T_WC_opt)
        
        R_opt = T_CW_opt[:3, :3]
        t_opt = T_CW_opt[:3, 3]
        
        rvec_opt, _ = cv2.Rodrigues(R_opt)
        
        kf['rvec'] = rvec_opt
        kf['tvec'] = t_opt

    print("BA: Finished.")