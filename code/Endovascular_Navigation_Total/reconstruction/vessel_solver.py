import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class VesselReconstructor:
    """
    Projection-constrained 3D reconstruction aligned with the original core solver.

    Inputs are image-space guidewire centerline points or particle centroids.
    Outputs are registered vessel/world coordinates in millimetres.
    """

    def __init__(self, config):
        self.config = config
        self.vessel_3d = self._load_vessel_world(
            config.VESSEL_NPY,
            config.NII_PATH,
            getattr(config, "VESSEL_NPY_COORDINATE", "world_mm"),
        )
        self.params, self.projection = self._load_registration(config.PARAM_JSON)
        self.cam_params = (
            self.params["tx"],
            self.params["ty"],
            self.params["tz"],
            self.params["pitch"],
            self.params["slew"],
            self.params["roll"],
        )
        self.sid = float(self.projection.get("SID", 1195.0))
        self.pix = float(self.projection.get("PIX", 0.184))
        self.num_ctrl_points = int(getattr(config, "RECON_CTRL_POINTS", 60))
        self.vessel_radius_mm = float(getattr(config, "RECON_VESSEL_RADIUS_MM", 3.5))
        self.w_proj = float(getattr(config, "RECON_W_PROJ", 1.0))
        self.w_smooth = float(getattr(config, "RECON_W_SMOOTH", 10.0))
        self.w_tube = float(getattr(config, "RECON_W_TUBE", 400.0))
        self.lambda_time = float(getattr(config, "RECON_LAMBDA_TIME", 5.0))
        self.tangent_points = int(getattr(config, "RECON_TANGENT_POINTS", 30))
        self.last_guidewire_3d = None
        self.last_particle_3d = None
        self.last_vessel_tangent_idx = None
        self.last_vessel_tangent = None
        self.last_vessel_match_idx = None

    def reconstruct_guidewire(self, centerline_2d, img_w, img_h):
        pts_2d_orig = np.asarray(centerline_2d, dtype=float)
        if pts_2d_orig.ndim != 2 or pts_2d_orig.shape[1] != 2 or len(pts_2d_orig) < 5:
            return None

        vessel_2d_proj = self.project(self.vessel_3d, img_w, img_h)
        if len(pts_2d_orig) > self.num_ctrl_points:
            idx = np.linspace(0, len(pts_2d_orig) - 1, self.num_ctrl_points).astype(int)
            target_2d = pts_2d_orig[idx]
        else:
            target_2d = pts_2d_orig

        is_cold_start = self.last_guidewire_3d is None
        tip_2d, tail_2d = target_2d[0], target_2d[-1]
        dists_2d_tip = np.linalg.norm(vessel_2d_proj - tip_2d, axis=1)
        dists_2d_tail = np.linalg.norm(vessel_2d_proj - tail_2d, axis=1)

        if is_cold_start:
            idx_tip = int(np.argmin(dists_2d_tip))
            idx_tail = int(np.argmin(dists_2d_tail))
            opt_maxiter = 500
        else:
            prev_tip_3d = self.last_guidewire_3d[0]
            prev_tail_3d = self.last_guidewire_3d[-1]
            dists_3d_tip = np.linalg.norm(self.vessel_3d - prev_tip_3d, axis=1)
            dists_3d_tail = np.linalg.norm(self.vessel_3d - prev_tail_3d, axis=1)
            idx_tip = int(np.argmin(dists_2d_tip + self.lambda_time * dists_3d_tip))
            idx_tail = int(np.argmin(dists_2d_tail + self.lambda_time * dists_3d_tail))
            opt_maxiter = 50

        lo, hi = sorted([idx_tip, idx_tail])
        if hi - lo < 5:
            lo = max(0, lo - 20)
            hi = min(len(self.vessel_3d) - 1, hi + 20)

        init_guidewire_3d = resample_curve(self.vessel_3d[lo:hi + 1], self.num_ctrl_points)

        def cost(flat_pts):
            pts_3d = flat_pts.reshape(-1, 3)
            u, v = self.project_components(pts_3d, img_w, img_h)
            err_proj = np.mean((u - target_2d[:, 0]) ** 2 + (v - target_2d[:, 1]) ** 2)
            if len(pts_3d) > 2:
                d2 = pts_3d[:-2] - 2 * pts_3d[1:-1] + pts_3d[2:]
                err_smooth = np.mean(np.sum(d2 * d2, axis=1))
            else:
                err_smooth = 0.0
            min_dists = np.min(cdist(pts_3d, self.vessel_3d), axis=1)
            err_tube = np.mean(np.maximum(0.0, min_dists - self.vessel_radius_mm) ** 2)
            return self.w_proj * err_proj + self.w_smooth * err_smooth + self.w_tube * err_tube

        result = minimize(
            cost,
            init_guidewire_3d.flatten(),
            method="SLSQP",
            options={"maxiter": opt_maxiter, "disp": False, "ftol": 1e-1},
        )
        guidewire_3d = result.x.reshape(-1, 3)
        self.last_cost = float(result.fun)
        if result.fun > 300.0 and not is_cold_start:
            self.last_guidewire_3d = None
            return None
        self.last_guidewire_3d = guidewire_3d.copy()
        return guidewire_3d

    def reconstruct_particle(self, center_2d, img_w, img_h):
        center = np.asarray(center_2d, dtype=float).reshape(2)
        vessel_2d = self.project(self.vessel_3d, img_w, img_h)
        dist_2d = np.linalg.norm(vessel_2d - center, axis=1)
        if self.last_particle_3d is not None:
            idx_axis = np.arange(len(self.vessel_3d), dtype=float)
            prev_idx = int(np.argmin(np.linalg.norm(self.vessel_3d - self.last_particle_3d, axis=1)))
            continuity = np.abs(idx_axis - float(prev_idx))
            backward = np.maximum(0.0, float(prev_idx) - idx_axis)
            idx = int(np.argmin(dist_2d + 0.02 * self.lambda_time * continuity + 0.1 * self.lambda_time * backward))
        else:
            idx = int(np.argmin(dist_2d))
        self.last_particle_3d = self.vessel_3d[idx].copy()
        return self.last_particle_3d

    def match_vessel_centerline(self, point_3d_mm, allow_backward=False):
        point = np.asarray(point_3d_mm, dtype=float).reshape(3)
        distances = np.linalg.norm(self.vessel_3d - point, axis=1)
        idx_axis = np.arange(len(self.vessel_3d), dtype=float)
        if self.last_vessel_match_idx is not None:
            prev = float(self.last_vessel_match_idx)
            continuity = np.abs(idx_axis - prev)
            distances = distances + 0.02 * self.lambda_time * continuity
            if not allow_backward:
                distances = distances + 0.1 * self.lambda_time * np.maximum(0.0, prev - idx_axis)
        idx = int(np.argmin(distances))
        self.last_vessel_match_idx = idx
        return idx, self.vessel_3d[idx]

    def vessel_target_near(self, point_3d_mm, lookahead_points=0):
        idx, _ = self.match_vessel_centerline(point_3d_mm)
        target_idx = min(len(self.vessel_3d) - 1, idx + max(0, int(lookahead_points)))
        return self.vessel_3d[target_idx].copy()

    def vessel_tangent_near(self, point_3d_mm, n=None):
        idx, _ = self.match_vessel_centerline(point_3d_mm)
        self.last_vessel_tangent_idx = idx

        count = int(n or self.tangent_points)
        half = max(1, count // 2)
        lo = max(0, idx - half)
        hi = min(len(self.vessel_3d), idx + half + 1)
        segment = self.vessel_3d[lo:hi]
        tangent = tangent_from_curve(segment, n=len(segment))
        if self.last_vessel_tangent is not None and np.dot(tangent, self.last_vessel_tangent) < 0:
            tangent = -tangent
        self.last_vessel_tangent = tangent.copy()
        return tangent

    def project(self, pts_3d, img_w, img_h):
        u, v = self.project_components(pts_3d, img_w, img_h)
        return np.column_stack([u, v])

    def project_components(self, pts_3d, img_w, img_h):
        tx, ty, tz, pitch, slew, roll = self.cam_params
        rot = get_rotation_matrix(pitch, slew, roll)
        pts_moved = (rot @ np.asarray(pts_3d, dtype=float).T).T + np.array([tx, ty, tz])
        dist = np.abs(pts_moved[:, 2])
        dist[dist < 1.0] = 1.0
        u = (pts_moved[:, 0] / dist) * (self.sid / self.pix) + img_w / 2.0
        v = (pts_moved[:, 1] / dist) * (self.sid / self.pix) + img_h / 2.0
        return u, v

    @staticmethod
    def _load_vessel_world(vessel_npy_path, nii_path, coordinate_mode="world_mm"):
        vessel_raw = np.load(vessel_npy_path).astype(float)
        if vessel_raw.shape[1] != 3:
            vessel_raw = vessel_raw.T
        coordinate_mode = str(coordinate_mode).lower()
        if coordinate_mode in ("world", "world_mm", "global", "global_mm"):
            return vessel_raw
        if coordinate_mode not in ("voxel", "index", "ijk"):
            raise ValueError("VESSEL_NPY_COORDINATE must be 'world_mm' or 'voxel'")
        img_nii = nib.load(nii_path)
        affine = img_nii.affine
        ones = np.ones((vessel_raw.shape[0], 1))
        return (affine @ np.hstack([vessel_raw, ones]).T)[:3, :].T

    @staticmethod
    def _load_registration(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["params"], data.get("projection", {})


def get_rotation_matrix(pitch, slew, roll):
    p, s, r = np.deg2rad([pitch, slew, roll])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(s), -np.sin(s)], [0, np.sin(s), np.cos(s)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def resample_curve(points, n):
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return points
    if len(points) < 4:
        idx = np.linspace(0, len(points) - 1, n).astype(int)
        return points[idx]
    try:
        tck, _ = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0, k=min(3, len(points) - 1))
        x, y, z = splev(np.linspace(0, 1, n), tck)
        return np.column_stack([x, y, z])
    except Exception:
        idx = np.linspace(0, len(points) - 1, n).astype(int)
        return points[idx]


def tangent_from_curve(points, n=30):
    points = np.asarray(points, dtype=float)
    segment = points[-min(n, len(points)):]
    if len(segment) < 2:
        return np.array([1.0, 0.0, 0.0])
    centered = segment - segment.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    if np.dot(direction, segment[-1] - segment[0]) < 0:
        direction = -direction
    return direction / max(np.linalg.norm(direction), 1e-12)
