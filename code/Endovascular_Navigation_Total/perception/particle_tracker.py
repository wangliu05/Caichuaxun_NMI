import cv2
import numpy as np


class ParticleTracker:
    def __init__(self, min_area_px=3):
        self.min_area_px = min_area_px

    def process(self, detection):
        mask = (detection["mask"] > 0).astype(np.uint8)
        area = float(mask.sum())
        if area < self.min_area_px:
            return None
        center = self._center(mask, detection["bbox_xyxy"])
        return {
            "type": "particle",
            "mask": mask,
            "center_2d": center,
            "area_px": area,
            "bbox_xyxy": detection["bbox_xyxy"],
            "confidence": detection["confidence"],
            "class_id": detection["class_id"],
            "class_name": detection["class_name"],
        }

    @staticmethod
    def _center(mask, bbox_xyxy):
        m = cv2.moments(mask, binaryImage=True)
        if abs(m["m00"]) > 1e-9:
            return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def draw_particle_overlay(image_bgr, target):
    center = target.get("center_2d")
    if center is None:
        return image_bgr
    c = tuple(np.round(center).astype(int))
    cv2.circle(image_bgr, c, 7, (0, 255, 0), 2)
    return image_bgr

