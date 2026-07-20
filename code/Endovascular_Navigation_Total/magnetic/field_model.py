import numpy as np

from .array import MU0, moments_from_q


class DipoleFieldModel:
    def __init__(self, magnet_array, min_distance_m=1e-4):
        self.array = magnet_array
        self.min_distance_m = min_distance_m
        self._hole_points, self._hole_dv = self._build_hole_quadrature()

    def field(self, p_m, q):
        p = np.asarray(p_m, dtype=float).reshape(3)
        moments, rotations = moments_from_q(self.array, q)
        total = np.zeros(3, dtype=float)
        for center, moment, rot in zip(self.array.centers_m, moments, rotations):
            total += dipole_field(p - center, moment, self.min_distance_m)
            if self.array.include_hole_correction and self._hole_points is not None:
                total -= self._hole_field(p, center, rot)
        return total

    def gradient(self, p_m, q):
        p = np.asarray(p_m, dtype=float).reshape(3)
        moments, rotations = moments_from_q(self.array, q)
        total = np.zeros((3, 3), dtype=float)
        for center, moment, rot in zip(self.array.centers_m, moments, rotations):
            total += dipole_gradient(p - center, moment, self.min_distance_m)
            if self.array.include_hole_correction and self._hole_points is not None:
                total -= self._hole_gradient(p, center, rot)
        return total

    def fields(self, points_m, q):
        return np.asarray([self.field(p, q) for p in np.asarray(points_m, dtype=float)])

    def gradient_magnitude(self, p_m, q):
        return float(np.linalg.norm(self.gradient(p_m, q), ord="fro"))

    def _build_hole_quadrature(self):
        if not self.array.include_hole_correction or self.array.hole_side_m <= 0:
            return None, 0.0
        n = max(1, int(self.array.hole_samples_per_axis))
        half = self.array.hole_side_m * 0.5
        xs = cell_centers(-half, half, n)
        ys = cell_centers(-half, half, n)
        zs = cell_centers(-self.array.radius_m, self.array.radius_m, n * 3)
        pts = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)
        pts = pts[np.linalg.norm(pts, axis=1) <= self.array.radius_m]
        volume = self.array.hole_side_m ** 2 * (2.0 * self.array.radius_m)
        dv = volume / max(len(pts), 1)
        return pts, dv

    def _hole_field(self, p, center, rot):
        m_local_density = np.array([0.0, 0.0, self.array.magnetization_a_per_m])
        total_local = np.zeros(3)
        rho = rot.T @ (p - center)
        for xi in self._hole_points:
            moment = m_local_density * self._hole_dv
            total_local += dipole_field(rho - xi, moment, self.min_distance_m)
        return rot @ total_local

    def _hole_gradient(self, p, center, rot):
        eps = 1e-4
        g = np.zeros((3, 3))
        for j in range(3):
            step = np.zeros(3)
            step[j] = eps
            g[:, j] = (self._hole_field(p + step, center, rot) - self._hole_field(p - step, center, rot)) / (2.0 * eps)
        return g


def dipole_field(r, m, min_distance_m=1e-4):
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    norm = max(float(np.linalg.norm(r)), min_distance_m)
    r_dot_m = float(np.dot(r, m))
    return MU0 / (4.0 * np.pi) * (3.0 * r * r_dot_m / norm ** 5 - m / norm ** 3)


def dipole_gradient(r, m, min_distance_m=1e-4):
    r = np.asarray(r, dtype=float)
    m = np.asarray(m, dtype=float)
    norm = max(float(np.linalg.norm(r)), min_distance_m)
    r_dot_m = float(np.dot(r, m))
    eye = np.eye(3)
    term1 = 3.0 * (eye * r_dot_m + np.outer(r, m)) / norm ** 5
    term2 = -15.0 * np.outer(r, r) * r_dot_m / norm ** 7
    term3 = 3.0 * np.outer(m, r) / norm ** 5
    return MU0 / (4.0 * np.pi) * (term1 + term2 + term3)


def cell_centers(start, stop, count):
    count = max(1, int(count))
    edges = np.linspace(float(start), float(stop), count + 1)
    return 0.5 * (edges[:-1] + edges[1:])
