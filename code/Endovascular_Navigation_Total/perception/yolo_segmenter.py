import time

import cv2
import numpy as np
from ultralytics import YOLO

from .types import normalize_target_name


class YoloSegmenter:
    def __init__(self, model_path, conf_thres=0.5, imgsz=1024, max_det=4, device="cpu"):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.model = YOLO(model_path, task="segment")
        self.names = getattr(self.model, "names", {}) or {}

    def predict(self, frame_bgr):
        t0 = time.perf_counter()
        kwargs = {
            "imgsz": self.imgsz,
            "retina_masks": True,
            "conf": self.conf_thres,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(frame_bgr, **kwargs)
        yolo_ms = (time.perf_counter() - t0) * 1000.0
        return {"detections": self._normalize(results[0], frame_bgr.shape[:2]), "timing": {"yolo_ms": yolo_ms}}

    def _normalize(self, result, frame_hw):
        h, w = frame_hw
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy() if getattr(result, "masks", None) is not None else None

        detections = []
        for i, box in enumerate(xyxy):
            class_id = int(cls[i])
            raw_name = self.names.get(class_id, str(class_id))
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": normalize_target_name(raw_name),
                    "raw_class_name": raw_name,
                    "confidence": float(conf[i]),
                    "mask": self._mask_for_detection(masks, i, box, w, h),
                    "bbox_xyxy": [float(v) for v in box],
                }
            )
        return detections

    @staticmethod
    def _mask_for_detection(masks, index, box, width, height):
        if masks is not None and index < len(masks):
            mask = masks[index]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            return (mask > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = np.round(box).astype(int)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        mask = np.zeros((height, width), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

