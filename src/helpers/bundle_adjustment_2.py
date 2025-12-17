import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# -------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3))
    return R


def project_point(X, rvec, tvec, K):
    R = rodrigues_to_R(rvec)
    Xc = R @ X + tvec
    if Xc[2] <= 1e-6:
        return None
    x = K @ Xc
    return x[:2] / x[2]


# -------------------------------------------------------------
# Bundle Adjustment residuals (camera 0 fixed)
# -------------------------------------------------------------

def ba_residuals(params, n_cams, n_points,
                 cam_indices, point_indices,
                 observations, K,
                 rvec0, tvec0):

    rvecs = np.zeros((n_cams, 3))
    tvecs = np.zeros((n_cams, 3))

    rvecs[0] = rvec0
    tvecs[0] = tvec0

    rvecs[1:] = params[:3*(n_cams-1)].reshape(-1, 3)
    tvecs[1:] = params[3*(n_cams-1):6*(n_cams-1)].reshape(-1, 3)

    points = params[6*(n_cams-1):].reshape(n_points, 3)

    residuals = np.zeros(len(observations) * 2, dtype=np.float64)

    for i, (ci, pi) in enumerate(zip(cam_indices, point_indices)):
        proj = project_point(points[pi], rvecs[ci], tvecs[ci], K)
        if proj is None:
            residuals[2*i:2*i+2] = 0
        else:
            residuals[2*i:2*i+2] = observations[i] - proj

    return residuals


# -------------------------------------------------------------
# Jacobian sparsity
# -------------------------------------------------------------

def ba_sparsity(n_cams, n_points, cam_indices, point_indices):
    m = len(cam_indices) * 2
    n = 6 * (n_cams - 1) + 3 * n_points
    A = lil_matrix((m, n), dtype=int)

    for i, (ci, pi) in enumerate(zip(cam_indices, point_indices)):
        r = 2 * i
        if ci > 0:
            A[r:r+2, 6*(ci-1):6*(ci-1)+6] = 1
        A[r:r+2, 6*(n_cams-1) + 3*pi:6*(n_cams-1) + 3*pi + 3] = 1

    return A


# -------------------------------------------------------------
# Local Bundle Adjustment
# -------------------------------------------------------------

def run_local_bundle_adjustment(rvecs, tvecs,
                                points_3d,
                                observations,
                                cam_indices,
                                point_indices,
                                K):

    n_cams = len(rvecs)
    n_points = len(points_3d)

    rvec0 = rvecs[0].copy()
    tvec0 = tvecs[0].copy()

    x0 = np.hstack([
        rvecs[1:].ravel(),
        tvecs[1:].ravel(),
        points_3d.ravel()
    ])

    sparsity = ba_sparsity(n_cams, n_points, cam_indices, point_indices)

    res = least_squares(
        ba_residuals,
        x0,
        jac_sparsity=sparsity,
        loss='huber',
        f_scale=3.0,
        ftol=1e-4,
        x_scale='jac',
        method='trf',
        verbose=0,
        args=(n_cams, n_points,
              cam_indices, point_indices,
              observations, K,
              rvec0, tvec0)
    )

    rvecs_opt = rvecs.copy()
    tvecs_opt = tvecs.copy()

    rvecs_opt[1:] = res.x[:3*(n_cams-1)].reshape(-1, 3)
    tvecs_opt[1:] = res.x[3*(n_cams-1):6*(n_cams-1)].reshape(-1, 3)

    points_opt = res.x[6*(n_cams-1):].reshape(n_points, 3)

    return rvecs_opt, tvecs_opt, points_opt


# -------------------------------------------------------------
# Build BA problem safely
# -------------------------------------------------------------

def build_ba_problem(map_points, keyframe_history):

    cam_id_to_idx = {kf['id']: i for i, kf in enumerate(keyframe_history)}

    rvecs = np.array([kf['rvec'] for kf in keyframe_history])
    tvecs = np.array([kf['tvec'] for kf in keyframe_history])

    points_3d = []
    observations = []
    cam_indices = []
    point_indices = []
    ba_point_ids = []

    for mp_idx, mp in enumerate(map_points):
        if len(mp.observations) < 3:
            continue

        ba_point_ids.append(mp_idx)
        points_3d.append(mp.position.copy())

        for fid, uv in mp.observations.items():
            if fid not in cam_id_to_idx:
                continue
            cam_indices.append(cam_id_to_idx[fid])
            point_indices.append(len(points_3d) - 1)
            observations.append(np.array(uv, dtype=np.float64))

    if len(observations) < 30:
        return None

    return (
        rvecs,
        tvecs,
        np.array(points_3d),
        np.array(observations),
        np.array(cam_indices),
        np.array(point_indices),
        ba_point_ids
    )


# -------------------------------------------------------------
# How to APPLY BA results (CRITICAL)
# -------------------------------------------------------------

def apply_ba_results(map_points, keyframe_history,
                     rvecs_opt, tvecs_opt,
                     points_opt, ba_point_ids):

    for kf, r, t in zip(keyframe_history, rvecs_opt, tvecs_opt):
        kf['rvec'][:] = r
        kf['tvec'][:] = t

    for idx, p in zip(ba_point_ids, points_opt):
        map_points[idx].position[:] = p