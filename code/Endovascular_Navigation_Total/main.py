import argparse
import time

import cv2
import numpy as np

from config import Config
from control import MagneticController
from core.actuator import create_actuator
from core.capture import create_frame_source
from magnetic import DipoleFieldModel, InverseActuationSolver, MagnetArray
from perception import PerceptionWorker
from reconstruction.vessel_solver import VesselReconstructor, tangent_from_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Fluoroscopy-guided magnetic navigation pipeline.")
    parser.add_argument("--mode", choices=["planar", "vascular_3d"], default=None)
    parser.add_argument("--target", choices=["guidewire", "particle", "both"], default=None)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until input ends or user exits.")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    if args.mode:
        config.EXPERIMENT_MODE = args.mode
    if args.target:
        config.TARGET_TYPE = args.target
    if args.show:
        config.SHOW_OVERLAY = True
    if args.no_show:
        config.SHOW_OVERLAY = False

    source = create_frame_source(config)
    source.open()

    perception = PerceptionWorker(config)
    controller = MagneticController(config)
    reconstructor = VesselReconstructor(config) if config.EXPERIMENT_MODE == "vascular_3d" else None

    magnet_array = MagnetArray.from_config(config)
    field_model = DipoleFieldModel(magnet_array)
    inverse_solver = InverseActuationSolver.from_config(field_model, config)
    actuator = create_actuator(config)

    window_name = "Endovascular Navigation Complete"
    if config.SHOW_OVERLAY:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    processed = 0
    try:
        while args.frames <= 0 or processed < args.frames:
            packet = source.read()
            if packet is None:
                if source.finished:
                    break
                continue

            t0 = time.perf_counter()
            perceived = perception.process_packet(packet)
            state = controller.visual_state_from_targets(perceived)
            if state is not None and reconstructor is not None:
                if state.target_type == "guidewire":
                    segment_mm = reconstructor.reconstruct_guidewire(state.centerline_2d, packet["width"], packet["height"])
                    if segment_mm is not None:
                        state.position_3d = segment_mm[-1] / 1000.0
                        state.tangent_3d = tangent_from_curve(segment_mm)
                        state.desired_tangent_3d = reconstructor.vessel_tangent_near(segment_mm[-1])
                elif state.target_type == "particle":
                    particle_mm = reconstructor.reconstruct_particle(state.position_2d, packet["width"], packet["height"])
                    if particle_mm is not None:
                        state.position_3d = particle_mm / 1000.0
                        target_mm = reconstructor.vessel_target_near(
                            particle_mm,
                            lookahead_points=config.PARTICLE_TARGET_LOOKAHEAD_POINTS,
                        )
                        state.raw["target_3d_m"] = target_mm / 1000.0

            objective = controller.objective_from_visual_state(state, perceived) if state is not None else None
            solve_result = None
            if objective is not None:
                solve_result = inverse_solver.solve(objective)
                if solve_result.success:
                    actuator.send(
                        inverse_solver.last_q,
                        metadata={"frame_idx": packet["frame_idx"], "objective": objective.kind, "config": config},
                    )

            total_ms = (time.perf_counter() - t0) * 1000.0
            print_status(perceived, state, objective, solve_result, total_ms)

            if config.SHOW_OVERLAY:
                cv2.imshow(window_name, perceived["overlay_bgr"])
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            processed += 1
    finally:
        source.close()
        if config.SHOW_OVERLAY:
            cv2.destroyAllWindows()


def print_status(packet, state, objective, solve_result, total_ms):
    timing = packet.get("timing", {})
    target_text = "none" if state is None else state.target_type
    objective_text = "none" if objective is None else objective.kind
    solve_text = "not-run"
    if solve_result is not None:
        solve_text = f"success={solve_result.success} cost={float(solve_result.fun):.3e}"
    print(
        f"[Frame {packet['frame_idx']}] target={target_text} objective={objective_text} "
        f"yolo={timing.get('yolo_ms', 0.0):.1f}ms "
        f"post={timing.get('postprocess_ms', 0.0):.1f}ms "
        f"loop={total_ms:.1f}ms {solve_text}"
    )


if __name__ == "__main__":
    main()
