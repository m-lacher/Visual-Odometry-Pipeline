import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# -----------------------------
# Geometry helpers
# -----------------------------

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R

def project_point(X, rvec, tvec, K):
    R = rodrigues_to_R(rvec)
    x = R @ X + tvec
    x = K @ x
    return x[:2] / x[2]

# -----------------------------
# Residuals (camera 0 fixed)
# -----------------------------

def ba_residuals(params, n_cams, n_points, cam_indices, point_indices, observations, K,
                 rvec0, tvec0):
    """
    Camera 0 is fixed to remove gauge freedom
    params = [rvecs(1..N) | tvecs(1..N) | points]
    """

    rvecs = np.zeros((n_cams, 3))
    tvecs = np.zeros((n_cams, 3))

    # unpack optimized cameras (1..N-1)
    rvecs[0] = rvec0
    tvecs[0] = tvec0

    rvecs[1:] = params[:3*(n_cams-1)].reshape((n_cams-1, 3))
    tvecs[1:] = params[3*(n_cams-1):6*(n_cams-1)].reshape((n_cams-1, 3))

    points = params[6*(n_cams-1):].reshape((n_points, 3))

    residuals = np.empty((len(observations) * 2,), dtype=np.float64)

    for i, (cam_i, pt_i, obs) in enumerate(zip(cam_indices, point_indices, observations)):
        proj = project_point(points[pt_i], rvecs[cam_i], tvecs[cam_i], K)
        residuals[2*i:2*i+2] = obs - proj

    return residuals

# -----------------------------
# Jacobian sparsity pattern
# -----------------------------

def ba_sparsity(n_cams, n_points, cam_indices, point_indices):
    """Sparse Jacobian for faster optimization"""
    m = len(cam_indices) * 2
    n = 6 * (n_cams - 1) + 3 * n_points
    A = lil_matrix((m, n), dtype=int)

    for i, (cam_i, pt_i) in enumerate(zip(cam_indices, point_indices)):
        row = 2 * i

        if cam_i > 0:
            cam_col = 6 * (cam_i - 1)
            A[row:row+2, cam_col:cam_col+6] = 1

        pt_col = 6 * (n_cams - 1) + 3 * pt_i
        A[row:row+2, pt_col:pt_col+3] = 1

    return A

# -----------------------------
# Bundle Adjustment Runner
# -----------------------------

def run_local_bundle_adjustment(
    rvecs,
    tvecs,
    points_3d,
    observations,
    cam_indices,
    point_indices,
    K
):
    n_cams = len(rvecs)
    n_points = len(points_3d)

    rvec0 = rvecs[0].copy()
    tvec0 = tvecs[0].copy()

    x0 = np.hstack([
        np.array(rvecs[1:]).ravel(),
        np.array(tvecs[1:]).ravel(),
        np.array(points_3d).ravel()
    ])

    sparsity = ba_sparsity(n_cams, n_points, cam_indices, point_indices)

    res = least_squares(
        ba_residuals,
        x0,
        jac_sparsity=sparsity,
        loss='huber',
        f_scale=5.0,
        verbose=2,
        x_scale='jac',
        ftol=1e-4,
        method='trf',
        args=(n_cams, n_points, cam_indices, point_indices, observations, K, rvec0, tvec0)
    )

    # unpack solution
    rvecs_opt = np.zeros_like(rvecs)
    tvecs_opt = np.zeros_like(tvecs)

    rvecs_opt[0] = rvec0
    tvecs_opt[0] = tvec0

    rvecs_opt[1:] = res.x[:3*(n_cams-1)].reshape((n_cams-1, 3))
    tvecs_opt[1:] = res.x[3*(n_cams-1):6*(n_cams-1)].reshape((n_cams-1, 3))

    points_opt = res.x[6*(n_cams-1):].reshape((n_points, 3))

    return rvecs_opt, tvecs_opt, points_opt, res

def build_ba_problem(map_points, keyframe_history):
    """
    Build BA inputs from map points and keyframes
    """

    cam_id_to_idx = {
        kf['id']: idx for idx, kf in enumerate(keyframe_history)
    }

    rvecs = []
    tvecs = []

    for kf in keyframe_history:
        rvecs.append(kf['rvec'].reshape(3))
        tvecs.append(kf['tvec'].reshape(3))

    points_3d = []
    observations = []
    cam_indices = []
    point_indices = []

    for pt_idx, mp in enumerate(map_points):
        for frame_id, uv in mp.observations.items():
            if frame_id not in cam_id_to_idx:
                continue

            cam_idx = cam_id_to_idx[frame_id]

            points_3d.append(mp.position)
            observations.append(uv)
            cam_indices.append(cam_idx)
            point_indices.append(pt_idx)

    if len(observations) < 20:
        return None

    return (
        np.array(rvecs),
        np.array(tvecs),
        np.array(points_3d),
        np.array(observations, dtype=np.float64),
        np.array(cam_indices),
        np.array(point_indices)
    )
