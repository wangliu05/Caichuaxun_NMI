import numpy as np
import time

from core.state import MagneticObjective, VisualState
from .mapper import ImageToWorkspaceMapper
from .path import PathMatcher, unit


class MagneticController:
    def __init__(self, config):
        self.config = config
        self.path_matcher = PathMatcher(config.PATH_TEMPLATE_JSON)
        self.mapper = ImageToWorkspaceMapper(
            config.WORKSPACE_RADIUS_M,
            center_pixel=getattr(config, "PLANAR_CENTER_PIXEL", None),
            center_global_mm=getattr(config, "PLANAR_CENTER_GLOBAL_MM", None),
            mm_per_pixel=getattr(config, "PLANAR_MM_PER_PIXEL", None),
            image_y_dir=getattr(config, "PLANAR_IMAGE_Y_DIR", -1.0),
        )
        self.guide_pid = GuidewireDirectionPid(
            kp=config.GUIDE_PID_KP,
            ki=config.GUIDE_PID_KI,
            kd=config.GUIDE_PID_KD,
            max_rot_rad=np.deg2rad(config.GUIDE_PID_MAX_ROT_DEG),
        )
        self.particle_pid = VectorPid(
            kp=config.PARTICLE_PID_KP,
            ki=config.PARTICLE_PID_KI,
            kd=config.PARTICLE_PID_KD,
            output_limit=config.PARTICLE_PID_OUTPUT_LIMIT_T_PER_M,
        )

    def visual_state_from_targets(self, packet):
        targets = packet.get("targets", [])
        if not targets:
            return None
        target = targets[0]
        if target["type"] == "guidewire":
            return VisualState(
                frame_idx=packet["frame_idx"],
                target_type="guidewire",
                position_2d=target["tip_2d"],
                tangent_2d=target["tip_tangent_2d"],
                centerline_2d=target["centerline_fit_2d"],
                raw=target,
            )
        if target["type"] == "particle":
            return VisualState(
                frame_idx=packet["frame_idx"],
                target_type="particle",
                position_2d=target["center_2d"],
                raw=target,
            )
        return None

    def objective_from_visual_state(self, visual_state, packet):
        if visual_state.target_type == "guidewire":
            return self._guidewire_objective(visual_state, packet)
        if visual_state.target_type == "particle":
            return self._particle_objective(visual_state, packet)
        return None

    def _guidewire_objective(self, state, packet):
        width = packet.get("width", 1)
        height = packet.get("height", 1)
        p_m = state.position_3d
        if p_m is None:
            p_m = self.mapper.point2d_to_workspace(state.position_2d, width, height)

        if state.tangent_3d is not None and state.desired_tangent_3d is not None:
            actual = unit(state.tangent_3d)
            desired = unit(state.desired_tangent_3d)
        else:
            actual = self.mapper.tangent2d_to_workspace(state.tangent_2d)
            desired_2d = self.path_matcher.desired_tangent_2d(state.position_2d, fallback=state.tangent_2d)
            desired = self.mapper.tangent2d_to_workspace(desired_2d)

        field_dir, error_rad = self.guide_pid.command_direction(actual, desired)
        b_desired = unit(field_dir) * float(self.config.GUIDE_FIELD_MAGNITUDE_T)
        return MagneticObjective(
            kind="guidewire_field",
            position_m=np.asarray(p_m, dtype=float),
            desired_field_t=b_desired,
            metadata={
                "frame_idx": state.frame_idx,
                "guide_error_rad": float(error_rad),
                "guide_error_deg": float(np.rad2deg(error_rad)),
            },
        )

    def _particle_objective(self, state, packet):
        width = packet.get("width", 1)
        height = packet.get("height", 1)
        p_m = state.position_3d
        if p_m is None:
            p_m = self.mapper.point2d_to_workspace(state.position_2d, width, height)

        target_m = None
        if isinstance(state.raw, dict):
            target_m = state.raw.get("target_3d_m")
        if target_m is None:
            target_2d = self.path_matcher.target_point_2d(
                state.position_2d,
                lookahead_points=self.config.PARTICLE_TARGET_LOOKAHEAD_POINTS,
                fallback=state.position_2d,
            )
            target_m = self.mapper.point2d_to_workspace(target_2d, width, height)
        target_m = np.asarray(target_m, dtype=float).reshape(3)
        error_m = target_m - np.asarray(p_m, dtype=float)
        if np.linalg.norm(error_m) * 1000.0 <= float(self.config.PARTICLE_TARGET_REACHED_MM):
            desired_drive = np.zeros(3, dtype=float)
        else:
            desired_drive = self.particle_pid.update(error_m)

        desired_dir = unit(desired_drive)
        if np.linalg.norm(desired_dir) < 1e-12:
            desired_dir = np.array([1.0, 0.0, 0.0])
        b_desired = desired_dir * float(self.config.PARTICLE_ALIGNMENT_FIELD_T)
        return MagneticObjective(
            kind="particle_gradient_drive",
            position_m=np.asarray(p_m, dtype=float),
            desired_field_t=b_desired,
            desired_gradient_drive_n=desired_drive,
            metadata={
                "frame_idx": state.frame_idx,
                "particle_error_m": error_m,
                "particle_error_mm": float(np.linalg.norm(error_m) * 1000.0),
                "target_m": target_m,
            },
        )


class GuidewireDirectionPid:
    def __init__(self, kp, ki, kd, max_rot_rad):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_rot_rad = abs(float(max_rot_rad))
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def command_direction(self, actual, desired):
        actual = unit(actual)
        desired = unit(desired)
        if np.linalg.norm(actual) < 1e-12:
            actual = desired
        if np.linalg.norm(desired) < 1e-12:
            desired = actual

        cross = np.cross(actual, desired)
        sin_theta = float(np.linalg.norm(cross))
        cos_theta = float(np.clip(np.dot(actual, desired), -1.0, 1.0))
        error = float(np.arctan2(sin_theta, cos_theta))
        if error < 1e-9:
            self.prev_error = error
            self.prev_time = time.monotonic()
            return desired, error

        now = time.monotonic()
        dt = 0.0 if self.prev_time is None else max(now - self.prev_time, 1e-6)
        self.prev_time = now
        derivative = 0.0 if self.prev_error is None or dt <= 0.0 else (error - self.prev_error) / dt
        self.prev_error = error
        self.integral += error * dt

        theta_cmd = self.kp * error + self.ki * self.integral + self.kd * derivative
        theta_cmd = float(np.clip(theta_cmd, 0.0, self.max_rot_rad))
        if sin_theta < 1e-9:
            axis = stable_perpendicular_axis(actual)
        else:
            axis = cross / sin_theta
        return rotate_about_axis(actual, axis, theta_cmd), error


class VectorPid:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = abs(float(output_limit))
        self.integral = np.zeros(3, dtype=float)
        self.prev_error = None
        self.prev_time = None

    def update(self, error):
        error = np.asarray(error, dtype=float).reshape(3)
        now = time.monotonic()
        dt = 0.0 if self.prev_time is None else max(now - self.prev_time, 1e-6)
        self.prev_time = now

        derivative = np.zeros(3, dtype=float)
        if self.prev_error is not None and dt > 0.0:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error.copy()
        self.integral += error * dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        norm = np.linalg.norm(output)
        if norm > self.output_limit > 0.0:
            output = output / norm * self.output_limit
        return output


def rotate_about_axis(vector, axis, angle_rad):
    v = unit(vector)
    a = unit(axis)
    angle = float(angle_rad)
    return (
        v * np.cos(angle)
        + np.cross(a, v) * np.sin(angle)
        + a * np.dot(a, v) * (1.0 - np.cos(angle))
    )


def stable_perpendicular_axis(vector):
    v = unit(vector)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(v, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    axis = np.cross(v, basis)
    return unit(axis)
