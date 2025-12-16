import numpy as np

def build_optimization_problem(map_points, keyframe_history):
    """
    Prepares data specifically for the provided runBA.py format.
    
    Structure of 'observations' expected by runBA:
    [Num_Frames, 
       Frame1_ID, Num_Obs, u1, v1, u2, v2..., LM_ID1, LM_ID2...,
       Frame2_ID, Num_Obs, u1, v1, u2, v2..., LM_ID1, LM_ID2...,
       ...
    ]
    """
    
    # 1. Select Active Map Points (seen by at least 2 frames in the window)
    window_frame_ids = [kf['id'] for kf in keyframe_history]
    window_frame_ids_set = set(window_frame_ids)
    
    active_map_points = []
    active_map_point_indices = [] # Indices in the global map_points list
    
    # Mapping from Global MapPoint Object to Local Index (0 to M)
    # runBA expects 1-based indices for landmarks!
    mp_global_to_local_id = {} 

    for global_idx, mp in enumerate(map_points):
        # Check intersection of observation keys and window frames
        if len(window_frame_ids_set.intersection(mp.observations.keys())) >= 2:
            active_map_points.append(mp)
            active_map_point_indices.append(global_idx)
            # Store 1-based index for runBA
            mp_global_to_local_id[mp] = len(active_map_points) 

    if not active_map_points:
        return None, None, None

    # 2. Build Hidden State (Poses + Landmarks)
    # Poses: [rx, ry, rz, tx, ty, tz]
    poses_flat = []
    for kf in keyframe_history:
        twist = np.concatenate((kf['rvec'].flatten(), kf['tvec'].flatten()))
        poses_flat.append(twist)
    
    # Landmarks: [x, y, z]
    points_flat = []
    for mp in active_map_points:
        points_flat.append(mp.position)
        
    hidden_state = np.concatenate((
        np.array(poses_flat).flatten(), 
        np.array(points_flat).flatten()
    ))
    
    # 3. Build Observations Vector (The Complex Serialized Part)
    obs_vector = []
    
    # First element: Number of frames
    obs_vector.append(len(keyframe_history))
    
    for kf in keyframe_history:
        frame_id = kf['id']
        
        # Collect valid observations for this specific frame
        frame_measurements = [] # Stores (u, v)
        frame_landmark_ids = [] # Stores local_landmark_id
        
        for mp in active_map_points:
            if frame_id in mp.observations:
                uv = mp.observations[frame_id]
                local_id = mp_global_to_local_id[mp]
                
                frame_measurements.extend([uv[0], uv[1]])
                frame_landmark_ids.append(local_id)
        
        # Append Frame Header
        # [Frame_ID, Num_Observations_In_This_Frame]
        obs_vector.append(frame_id) # This is 'observations[observation_i]'
        obs_vector.append(len(frame_landmark_ids)) # This is 'observations[observation_i + 1]'
        
        # Append Coordinates: u1, v1, u2, v2 ...
        obs_vector.extend(frame_measurements)
        
        # Append Landmark Indices: id1, id2 ...
        obs_vector.extend(frame_landmark_ids)

    observations = np.array(obs_vector, dtype=np.float32)
    
    mapping_data = {
        'active_map_point_indices': active_map_point_indices,
        'num_frames': len(keyframe_history)
    }
    
    return hidden_state, observations, mapping_data

def update_objects_from_state(optimized_state, map_points, keyframe_history, mapping_data):
    """
    Unpacks the optimized state and updates the objects in place.
    """
    num_frames = mapping_data['num_frames']
    mp_indices = mapping_data['active_map_point_indices']
    
    # 1. Unpack Poses
    poses_end_idx = num_frames * 6
    poses_flat = optimized_state[:poses_end_idx].reshape((num_frames, 6))
    
    for i, kf in enumerate(keyframe_history):
        kf['rvec'] = poses_flat[i, :3].reshape(3, 1)
        kf['tvec'] = poses_flat[i, 3:].reshape(3, 1)
        
    # 2. Unpack Landmarks
    landmarks_flat = optimized_state[poses_end_idx:].reshape((len(mp_indices), 3))
    
    for i, global_idx in enumerate(mp_indices):
        map_points[global_idx].position = landmarks_flat[i]