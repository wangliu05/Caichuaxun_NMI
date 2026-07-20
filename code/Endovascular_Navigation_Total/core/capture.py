import sys
import time
import traceback

import cv2
import numpy as np


class FrameSource:
    mode = "capture_card"
    finished = False

    def open(self):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def close(self):
        pass

    def period(self):
        return None


class CaptureCardFrameSource(FrameSource):
    mode = "capture_card"

    def __init__(self, config):
        self.config = config
        self.cap = None
        self.fourcc_map = None
        self.mwcap_module = None
        self.frame_idx = 0

    def open(self):
        try:
            sdk_path = getattr(self.config, "MAGEWELL_SDK_PATH", None)
            if sdk_path and sdk_path not in sys.path:
                sys.path.insert(0, sdk_path)
            import Capture as mwcap_module
            from Capture import Capture
            from LibMWCapture import MWFOURCC_BGR24, MWFOURCC_NV12, MWFOURCC_RGB24, MWFOURCC_YUY2
        except Exception as exc:
            traceback.print_exc()
            raise RuntimeError(
                f"Magewell SDK import failed: {exc}. Check MAGEWELL_SDK_PATH and capture-card SDK installation."
            ) from exc

        self.mwcap_module = mwcap_module
        self.fourcc_map = {
            "NV12": MWFOURCC_NV12,
            "YUY2": MWFOURCC_YUY2,
            "RGB24": MWFOURCC_RGB24,
            "BGR24": MWFOURCC_BGR24,
        }
        fourcc = str(getattr(self.config, "FOURCC", "NV12")).upper()
        if fourcc not in self.fourcc_map:
            raise ValueError(f"Unsupported FOURCC: {fourcc}")

        fps = max(1, int(round(float(getattr(self.config, "TARGET_FPS", 1)))))
        self.mwcap_module.CAPTURE_FRAME_RATE = fps

        self.cap = Capture()
        self.cap.set_video(
            self.fourcc_map[fourcc],
            int(getattr(self.config, "CAP_WIDTH", 1920)),
            int(getattr(self.config, "CAP_HEIGHT", 1080)),
        )
        self.cap.set_audio(channels=2, sample_rate=48000, bit_per_sample=16)
        if self.cap.start_capture(int(getattr(self.config, "DEVICE_INDEX", 0))) != 0:
            raise RuntimeError("Magewell start_capture failed")
        print(f"[Capture] capture card started: device={getattr(self.config, 'DEVICE_INDEX', 0)} fourcc={fourcc}")

    def read(self):
        if self.cap is None:
            return None

        last_buf, last_ts = None, 0
        while True:
            buf, ts = self.cap.get_video_rec()
            if buf == 0:
                break
            last_buf, last_ts = buf, ts

        if last_buf is None:
            return None

        cap_width = int(getattr(self.config, "CAP_WIDTH", 1920))
        cap_height = int(getattr(self.config, "CAP_HEIGHT", 1080))
        fourcc = str(getattr(self.config, "FOURCC", "NV12")).upper()
        bgr = buf_to_bgr(last_buf, fourcc, cap_width, cap_height, self.cap.m_min_stride)
        meta = {"device_index": int(getattr(self.config, "DEVICE_INDEX", 0))}
        frame = maybe_apply_roi(bgr, self.config, meta, True)
        packet = frame_packet(frame, self.frame_idx, last_ts, self.mode, meta)
        self.frame_idx += 1
        return packet

    def close(self):
        if self.cap is not None:
            self.cap.stop_capture()
            self.cap = None
            print("[Capture] capture card stopped")


def create_frame_source(config):
    return CaptureCardFrameSource(config)


def frame_packet(frame_bgr, frame_idx, timestamp, source, meta):
    h, w = frame_bgr.shape[:2]
    return {
        "frame_bgr": frame_bgr,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "source": source,
        "width": w,
        "height": h,
        "meta": meta,
    }


def maybe_apply_roi(frame, config, meta, apply_roi=None):
    if apply_roi is None:
        apply_roi = getattr(config, "APPLY_ROI", False)
    if not apply_roi:
        meta["roi"] = None
        meta["coordinate_space"] = "full_frame"
        return frame
    h, w = frame.shape[:2]
    rx = max(0, min(int(config.ROI_X), w - 1))
    ry = max(0, min(int(config.ROI_Y), h - 1))
    rw = int(config.ROI_W) if int(config.ROI_W) > 0 else w - rx
    rh = int(config.ROI_H) if int(config.ROI_H) > 0 else h - ry
    rw = max(1, min(rw, w - rx))
    rh = max(1, min(rh, h - ry))
    meta["roi"] = {"x": rx, "y": ry, "w": rw, "h": rh}
    meta["original_width"] = w
    meta["original_height"] = h
    meta["coordinate_space"] = "roi"
    return frame[ry:ry + rh, rx:rx + rw].copy()


def buf_to_bgr(raw_buf, fourcc_name, width, height, stride):
    fourcc_name = str(fourcc_name).upper()
    if fourcc_name == "NV12":
        true_size = stride * height * 3 // 2
        raw = np.frombuffer(raw_buf, dtype=np.uint8, count=true_size)
        yuv = raw.reshape((height * 3 // 2, stride))[:, :width]
        return cv2.cvtColor(np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_NV12)

    if fourcc_name == "YUY2":
        raw = np.frombuffer(raw_buf, dtype=np.uint8, count=stride * height)
        yuy2 = raw.reshape((height, stride))[:, :width * 2]
        return cv2.cvtColor(np.ascontiguousarray(yuy2).reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUY2)

    if fourcc_name in ("RGB24", "BGR24"):
        raw = np.frombuffer(raw_buf, dtype=np.uint8, count=stride * height)
        img = raw.reshape((height, stride))[:, :width * 3]
        img = np.ascontiguousarray(img).reshape((height, width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if fourcc_name == "RGB24" else img

    raise ValueError(f"Unsupported FOURCC: {fourcc_name}")


def run_capture_loop(config, on_frame, stop_event=None):
    source = create_frame_source(config)
    source.open()
    period = source.period() or 1.0 / max(float(config.TARGET_FPS), 1e-6)
    next_time = time.monotonic()
    try:
        while stop_event is None or not stop_event.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.005, next_time - now))
                continue
            next_time += period
            packet = source.read()
            if packet is None:
                if source.finished:
                    break
                continue
            on_frame(packet)
    finally:
        source.close()
