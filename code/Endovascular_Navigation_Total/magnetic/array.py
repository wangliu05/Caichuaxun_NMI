from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * np.pi


@dataclass
class MagnetArray:
    centers_m: np.ndarray
    radius_m: float
    remanence_t: float
    phi_rad: np.ndarray
    polarity_z: np.ndarray
    hole_side_m: float = 0.0
    include_hole_correction: bool = False
    hole_samples_per_axis: int = 3

    @property
    def magnet_count(self):
        return int(self.centers_m.shape[0])

    @property
    def solid_volume_m3(self):
        return 4.0 / 3.0 * np.pi * self.radius_m ** 3

    @property
    def magnetization_a_per_m(self):
        return self.remanence_t / MU0

    @property
    def dipole_moment_magnitude(self):
        return self.magnetization_a_per_m * self.solid_volume_m3

    @classmethod
    def from_config(cls, config):
        centers = fig3g_centers(
            radius_m=config.MAGNET_RADIUS_M,
            workspace_radius_m=config.WORKSPACE_RADIUS_M,
            tol_distance_m=config.MAGNET_TOL_DISTANCE_M,
            alpha_deg=config.MAGNET_ALPHA_DEG,
            layer_heights_m=config.MAGNET_LAYER_HEIGHTS_M,
        )
        return cls(
            centers_m=centers,
            radius_m=config.MAGNET_RADIUS_M,
            remanence_t=config.MAGNET_REMANCE_T,
            phi_rad=np.deg2rad(np.asarray(config.MAGNET_PHI_DEG, dtype=float)),
            polarity_z=np.asarray(config.MAGNET_POLARITY_Z, dtype=float),
            hole_side_m=config.HOLE_SIDE_M,
            include_hole_correction=config.INCLUDE_HOLE_CORRECTION,
            hole_samples_per_axis=config.HOLE_SAMPLES_PER_AXIS,
        )


def fig3g_centers(radius_m, workspace_radius_m, tol_distance_m, alpha_deg=52.0, layer_heights_m=None):
    radial_center = workspace_radius_m + radius_m
    alpha = np.deg2rad(alpha_deg)
    rho = -np.pi / 4.0 + alpha
    ang_pos = np.array([
        rho,
        alpha + rho,
        -alpha + rho,
        np.pi - rho,
        np.pi - alpha - rho,
        np.pi + alpha - rho,
    ], dtype=float)

    if layer_heights_m is None:
        layer_offset = 2.0 * radius_m + tol_distance_m
        z_values = np.array([0.0, layer_offset, -layer_offset], dtype=float)
    else:
        z_values = np.asarray(layer_heights_m, dtype=float)

    centers = []
    for z in z_values:
        for angle in ang_pos:
            x = radial_center * np.cos(angle)
            y = radial_center * np.sin(angle)
            centers.append([x, y, z])
    return np.asarray(centers, dtype=float)


def rotation_from_angles(phi, gamma, beta):
    cp, sp = np.cos(phi), np.sin(phi)
    cg, sg = np.cos(gamma), np.sin(gamma)
    cb, sb = np.cos(beta), np.sin(beta)
    rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    ry = np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
    return rz @ rx @ ry


def moments_from_q(array, q):
    q = np.asarray(q, dtype=float).reshape(array.magnet_count, 2)
    local_axis = np.array([0.0, 0.0, 1.0])
    moments = []
    rotations = []
    for idx, (gamma, beta) in enumerate(q):
        rot = rotation_from_angles(array.phi_rad[idx], gamma, beta)
        rotations.append(rot)
        moments.append(array.dipole_moment_magnitude * array.polarity_z[idx] * (rot @ local_axis))
    return np.asarray(moments), np.asarray(rotations)
