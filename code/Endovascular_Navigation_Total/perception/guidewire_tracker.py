from collections import deque

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize


class GuidewireTracker:
    def __init__(self, min_points=8, tangent_points=30, smooth_samples=180):
        self.min_points = min_points
        self.tangent_points = tangent_points
        self.smooth_samples = smooth_samples

    def process(self, detection):
        mask = (detection["mask"] > 0).astype(np.uint8)
        skeleton = skeletonize(mask > 0).astype(np.uint8)
        centerline = self._longest_skeleton_path(skeleton)
        if len(centerline) < self.min_points:
            return None

        centerline_fit = self._smooth_and_densify(centerline)
        tip = centerline_fit[-1]
        tail = centerline_fit[0]
        tangent = self._tip_tangent(centerline_fit)
        return {
            "type": "guidewire",
            "mask": mask,
            "skeleton": skeleton,
            "centerline_2d": centerline.astype(np.float32),
            "centerline_fit_2d": centerline_fit.astype(np.float32),
            "tip_2d": tip.astype(np.float32),
            "tail_2d": tail.astype(np.float32),
            "tip_tangent_2d": tangent.astype(np.float32),
            "confidence": detection["confidence"],
            "bbox_xyxy": detection["bbox_xyxy"],
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
        }

    def _longest_skeleton_path(self, skeleton):
        ys, xs = np.nonzero(skeleton)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)

        pixels = {(int(x), int(y)) for x, y in zip(xs, ys)}
        components = self._components(pixels)
        component = max(components, key=len)
        if len(component) < self.min_points:
            return np.empty((0, 2), dtype=np.float32)

        start = next(iter(component))
        a, _ = self._farthest_from(start, component)
        b, parents = self._farthest_from(a, component)
        path = self._recover_path(a, b, parents)
        return np.asarray(path, dtype=np.float32)

    def _components(self, pixels):
        unvisited = set(pixels)
        components = []
        while unvisited:
            start = unvisited.pop()
            comp = {start}
            queue = deque([start])
            while queue:
                cur = queue.popleft()
                for nb in neighbors8(cur):
                    if nb in unvisited:
                        unvisited.remove(nb)
                        comp.add(nb)
                        queue.append(nb)
            components.append(comp)
        return components

    def _farthest_from(self, start, component):
        queue = deque([start])
        parents = {start: None}
        distance = {start: 0}
        farthest = start
        while queue:
            cur = queue.popleft()
            if distance[cur] > distance[farthest]:
                farthest = cur
            for nb in neighbors8(cur):
                if nb in component and nb not in parents:
                    parents[nb] = cur
                    distance[nb] = distance[cur] + 1
                    queue.append(nb)
        return farthest, parents

    @staticmethod
    def _recover_path(start, end, parents):
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            if cur == start:
                break
            cur = parents.get(cur)
        path.reverse()
        return path

    def _smooth_and_densify(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) < 4:
            return pts
        distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        if distances[-1] <= 1e-6:
            return pts
        keep = np.r_[True, np.diff(distances) > 1e-6]
        pts = pts[keep]
        k = min(3, len(pts) - 1)
        try:
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=max(1.0, len(pts) * 0.2), k=k)
            samples = max(self.smooth_samples, len(pts))
            x, y = splev(np.linspace(0, 1, samples), tck)
            return np.column_stack([x, y])
        except Exception:
            return pts

    def _tip_tangent(self, pts):
        segment = pts[-min(self.tangent_points, len(pts)):]
        if len(segment) < 2:
            return np.array([0.0, 0.0])
        centered = segment - segment.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        if np.dot(direction, segment[-1] - segment[0]) < 0:
            direction = -direction
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-9 else np.array([0.0, 0.0])


def neighbors8(point):
    x, y = point
    return [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
        (x - 1, y),                 (x + 1, y),
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1),
    ]


def draw_guidewire_overlay(image_bgr, target):
    overlay = image_bgr
    mask = target.get("mask")
    if mask is not None:
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)
    pts = np.asarray(target.get("centerline_fit_2d", []), dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, (0, 0, 255), 2)
    tip = target.get("tip_2d")
    if tip is not None:
        p = tuple(np.round(tip).astype(int))
        cv2.circle(overlay, p, 5, (0, 255, 255), -1)
    return overlay
