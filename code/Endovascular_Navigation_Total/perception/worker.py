import time

from .guidewire_tracker import GuidewireTracker, draw_guidewire_overlay
from .particle_tracker import ParticleTracker, draw_particle_overlay
from .types import normalize_target_name, target_matches
from .yolo_segmenter import YoloSegmenter


class PerceptionWorker:
    def __init__(self, config):
        self.config = config
        self.segmenter = YoloSegmenter(
            config.MODEL_PATH,
            conf_thres=config.YOLO_CONF_THRES,
            imgsz=config.YOLO_IMGSZ,
            max_det=config.YOLO_MAX_DETECTIONS,
            device=config.YOLO_DEVICE,
        )
        self.guidewire_tracker = GuidewireTracker()
        self.particle_tracker = ParticleTracker()

    def process_packet(self, packet):
        frame = packet["frame_bgr"]
        t0 = time.perf_counter()
        pred = self.segmenter.predict(frame)
        post_t0 = time.perf_counter()
        overlay = frame.copy()
        targets = []
        for det in pred["detections"]:
            target_type = self._resolve_detection_type(det)
            if target_type == "guidewire":
                target = self.guidewire_tracker.process(det)
                if target is not None:
                    targets.append(target)
                    overlay = draw_guidewire_overlay(overlay, target)
            elif target_type == "particle":
                target = self.particle_tracker.process(det)
                if target is not None:
                    targets.append(target)
                    overlay = draw_particle_overlay(overlay, target)
        timing = {
            "yolo_ms": pred["timing"]["yolo_ms"],
            "postprocess_ms": (time.perf_counter() - post_t0) * 1000.0,
            "total_ms": (time.perf_counter() - t0) * 1000.0,
        }
        return {**packet, "targets": targets, "overlay_bgr": overlay, "timing": timing}

    def _resolve_detection_type(self, det):
        target = self.config.TARGET_TYPE
        name = normalize_target_name(det.get("class_name"))
        if target_matches(name, target):
            return name
        if det.get("class_id") == self.config.GUIDEWIRE_CLASS_ID and target in ("guidewire", "both", "all"):
            return "guidewire"
        if det.get("class_id") == self.config.PARTICLE_CLASS_ID and target in ("particle", "both", "all"):
            return "particle"
        return None

