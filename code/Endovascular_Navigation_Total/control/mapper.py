import numpy as np


class ImageToWorkspaceMapper:
    def __init__(
        self,
        workspace_radius_m=0.12,
        center_pixel=None,
        center_global_mm=None,
        mm_per_pixel=None,
        image_y_dir=-1.0,
    ):
        self.workspace_radius_m = workspace_radius_m
        self.center_pixel = None if center_pixel is None else np.asarray(center_pixel, dtype=float).reshape(2)
        self.center_global_m = None if center_global_mm is None else np.asarray(center_global_mm, dtype=float).reshape(3) / 1000.0
        self.mm_per_pixel = None if mm_per_pixel is None else float(mm_per_pixel)
        self.image_y_dir = 1.0 if float(image_y_dir) >= 0 else -1.0

    def point2d_to_workspace(self, point_2d, width, height, z_m=0.0):
        point = np.asarray(point_2d, dtype=float)
        if self.center_pixel is not None and self.center_global_m is not None and self.mm_per_pixel is not None:
            delta_px = point - self.center_pixel
            delta_m = delta_px * (self.mm_per_pixel / 1000.0)
            return self.center_global_m + np.array([delta_m[0], self.image_y_dir * delta_m[1], 0.0], dtype=float)

        scale = 2.0 * self.workspace_radius_m / max(width, height, 1)
        x = (point[0] - width * 0.5) * scale
        y = (height * 0.5 - point[1]) * scale
        return np.array([x, y, z_m], dtype=float)

    def tangent2d_to_workspace(self, tangent_2d):
        t = np.asarray(tangent_2d, dtype=float)
        v = np.array([t[0], self.image_y_dir * t[1], 0.0], dtype=float)
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
