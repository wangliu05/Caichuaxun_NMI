import json
from pathlib import Path

import numpy as np


class PathMatcher:
    def __init__(self, path_json=None, continuity_weight=0.25):
        self.paths = []
        self.last_path_idx = None
        self.last_point_idx = None
        self.continuity_weight = float(continuity_weight)
        if path_json and Path(path_json).exists():
            with open(path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.paths = data.get("paths", [])

    def desired_tangent_2d(self, point_2d, fallback=None):
        match = self.match(point_2d)
        if match is None:
            return unit(fallback) if fallback is not None else np.array([1.0, 0.0])
        return unit(match["tangent"])

    def target_point_2d(self, point_2d, lookahead_points=1, fallback=None):
        match = self.match(point_2d)
        if match is None:
            return None if fallback is None else np.asarray(fallback, dtype=float)
        path = self.paths[match["path_idx"]]
        pts = np.asarray(path.get("sampled_points", []), dtype=float)
        target_idx = min(len(pts) - 1, match["point_idx"] + max(0, int(lookahead_points)))
        return pts[target_idx]

    def match(self, point_2d):
        if not self.paths:
            return None
        point = np.asarray(point_2d, dtype=float)
        best = None
        for path_idx, path in enumerate(self.paths):
            pts = np.asarray(path.get("sampled_points", []), dtype=float)
            tangents = np.asarray(path.get("tangents", []), dtype=float)
            if len(pts) == 0 or len(tangents) != len(pts):
                continue
            distances = np.linalg.norm(pts - point, axis=1)
            if self.last_path_idx == path_idx and self.last_point_idx is not None:
                idx_axis = np.arange(len(pts), dtype=float)
                distances = distances + self.continuity_weight * np.abs(idx_axis - float(self.last_point_idx))
            idx = int(np.argmin(distances))
            score = float(distances[idx])
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "path_idx": path_idx,
                    "point_idx": idx,
                    "point": pts[idx],
                    "tangent": tangents[idx],
                }
        if best is None:
            return None
        self.last_path_idx = best["path_idx"]
        self.last_point_idx = best["point_idx"]
        return best


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v
