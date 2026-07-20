from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = ROOT / "resources"


@dataclass
class Config:
    # Experiment mode: "planar" or "vascular_3d"
    EXPERIMENT_MODE: str = "planar"
    TARGET_TYPE: str = "guidewire"  # "guidewire", "particle", or "both"

    # Capture-card input
    TARGET_FPS: float = 5.0

    # Compatible with the original core/dsa_capture.py.
    DEVICE_INDEX: int = 0
    FOURCC: str = "NV12"
    CAP_WIDTH: int = 1920
    CAP_HEIGHT: int = 1080
    MAGEWELL_SDK_PATH: str = r"/home/dsa/APP/capture/SDK/Capture_SDK_Linux_3.3.1.1313/Examples/python/AVCapture"
    SAVE_FRAMES: bool = False
    RECORD_FOLDER_FRAMES: bool = False
    SAVE_DIR: str = str(ROOT / "output" / "captured_frames")

    # Optional ROI
    APPLY_ROI: bool = False
    ROI_X: int = 0
    ROI_Y: int = 0
    ROI_W: int = 0
    ROI_H: int = 0

    # YOLO
    MODEL_PATH: str = str(RESOURCE_ROOT / "models" / "best.onnx")
    YOLO_IMGSZ: int = 1024
    YOLO_CONF_THRES: float = 0.5
    YOLO_MAX_DETECTIONS: int = 4
    YOLO_DEVICE: str = "cpu"
    GUIDEWIRE_CLASS_ID: int = 0
    PARTICLE_CLASS_ID: int = 1
    SHOW_OVERLAY: bool = True

    # 3D reconstruction resources
    VESSEL_NPY: str = str(RESOURCE_ROOT / "vascular" / "centerline_segment2.npy")
    PARAM_JSON: str = str(RESOURCE_ROOT / "vascular" / "registration_params_manual_rabbit.json")
    NII_PATH: str = str(RESOURCE_ROOT / "vascular" / "vessel_final.nii.gz")
    STL_PATH: str = str(RESOURCE_ROOT / "vascular" / "vessel_world_smooth.stl")
    VESSEL_NPY_COORDINATE: str = "world_mm"  # "world_mm" or "voxel"
    RECON_TANGENT_POINTS: int = 30

    # Path tracking
    PATH_TEMPLATE_JSON: str = str(RESOURCE_ROOT / "paths" / "guidewire_2d_paths_pid_all_limited75.json")
    GUIDE_FIELD_MAGNITUDE_T: float = 0.040
    GUIDE_PID_KP: float = 1.0
    GUIDE_PID_KI: float = 0.0
    GUIDE_PID_KD: float = 0.05
    GUIDE_PID_MAX_ROT_DEG: float = 35.0
    # 2D planar calibration: global model center plus pixel-to-mm scale.
    PLANAR_CENTER_PIXEL: tuple = (960.0, 540.0)
    PLANAR_CENTER_GLOBAL_MM: tuple = (0.0, 0.0, 0.0)
    PLANAR_MM_PER_PIXEL: float = 0.184
    PLANAR_IMAGE_Y_DIR: float = -1.0

    PARTICLE_ALIGNMENT_FIELD_T: float = 0.015
    PARTICLE_PID_KP: float = 2.0
    PARTICLE_PID_KI: float = 0.0
    PARTICLE_PID_KD: float = 0.1
    PARTICLE_PID_OUTPUT_LIMIT_T_PER_M: float = 0.3
    PARTICLE_TARGET_LOOKAHEAD_POINTS: int = 12
    PARTICLE_TARGET_REACHED_MM: float = 2.0

    # Magnet array, SI units
    MAGNET_COUNT_PER_LAYER: int = 6
    MAGNET_LAYER_COUNT: int = 3
    MAGNET_RADIUS_M: float = 0.030
    MAGNET_REMANCE_T: float = 1.44
    WORKSPACE_RADIUS_M: float = 0.120
    WORKSPACE_HEIGHT_M: float = 0.300
    MAGNET_TOL_DISTANCE_M: float = 0.052
    MAGNET_ALPHA_DEG: float = 52.0
    MAGNET_LAYER_HEIGHTS_M: object = None
    MAGNET_PHI_DEG: tuple = (
        10, 30, 10, -10, -30, -10,
        10, 30, 10, -10, -30, -10,
        10, 30, 10, -10, -30, -10,
    )
    MAGNET_POLARITY_Z: tuple = (
        -1, -1, -1, 1, 1, 1,
        -1, -1, -1, 1, 1, 1,
        -1, -1, -1, 1, 1, 1,
    )
    INCLUDE_HOLE_CORRECTION: bool = True
    HOLE_SIDE_M: float = 0.007
    HOLE_SAMPLES_PER_AXIS: int = 1

    # Inverse actuation
    INVERSE_SOLVER: str = "auto"  # "auto", "casadi", or "scipy"
    INVERSE_MAXITER: int = 80
    INVERSE_LAMBDA_DQ: float = 1e-4
    INVERSE_MAX_STEP_RAD: float = 0.5235987756
    MIN_PARTICLE_FIELD_T: float = 0.015
    IPOPT_PRINT_LEVEL: int = 0

    # Actuator backend
    ACTUATOR_BACKEND: str = "mock"
    ACTUATOR_LOG: str = str(ROOT / "output" / "actuator_commands.csv")
    ACTUATOR_SAVE_LOG: bool = False
    ACTUATOR_USE_ENCODER_PID: bool = False

    # PCA9685/PCA9585-style I2C PWM servo backend.
    # If your board is the common 16-channel PWM servo driver, it is PCA9685.
    # The backend accepts "pca9685" and "pca9585" because PCA9585 is often used
    # as a local typo/name for the same motor-control board.
    PCA_I2C_BUS: int = 1
    PCA_I2C_ADDRESSES: tuple = (0x40, 0x41, 0x42)
    PCA_PWM_FREQUENCY_HZ: float = 50.0
    PCA_CHANNEL_MAP: object = None  # Optional tuple of 36 (board_index, channel_index) pairs.
    SERVO_MIN_PULSE_US: float = 500.0
    SERVO_MAX_PULSE_US: float = 2500.0
    SERVO_MIN_ANGLE_DEG: float = 0.0
    SERVO_MAX_ANGLE_DEG: float = 360.0
    SERVO_INVERTED_AXES: tuple = ()
    SERVO_BETA_INIT_DEG: tuple = (
        349, 141.5, 207, 170, 178, 17,
        183.5, 195, 152, 65, 341, 355,
        188, 204, 253, 152, 90.5, 250,
    )
    SERVO_GAMMA_INIT_DEG: tuple = (
        50, 115, 200, 275, 1, 320,
        170, 210, 270, 255, 170, 145,
        199, 180, 210, 200, 180, 70,
    )
    SERVO_GAMMA_DIR: tuple = field(default_factory=lambda: tuple([1] * 18))
    SERVO_BETA_DIR: tuple = field(default_factory=lambda: tuple([1] * 18))

    # Encoder PID settings. PCA9685 only generates PWM; it cannot read a grating
    # encoder by itself. FT-EPC-04 is read through RS485 Modbus-RTU.
    ENCODER_BACKEND: str = "none"
    ENCODER_SERIAL_PORT: str = "COM3"
    ENCODER_BAUDRATE: int = 9600
    ENCODER_BYTESIZE: int = 8
    ENCODER_PARITY: str = "N"
    ENCODER_STOPBITS: int = 1
    ENCODER_TIMEOUT_S: float = 0.08
    ENCODER_SLAVE_IDS: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    ENCODER_AXIS_MAP: object = None  # Optional tuple of 36 (slave_id, channel_0_to_3) pairs.
    ENCODER_COUNT_START_REGISTER: int = 0x0030
    ENCODER_COUNT_REGISTER_COUNT: int = 8
    ENCODER_WORD_ORDER: str = "high_low"
    ENCODER_COUNTS_PER_REV: float = 4096.0
    ENCODER_ZERO_COUNTS: tuple = field(default_factory=lambda: tuple([0] * 36))
    ENCODER_ZERO_ALPHA_DEG: tuple = field(default_factory=lambda: tuple([0.0] * 36))
    ENCODER_ALPHA_DIR: tuple = field(default_factory=lambda: tuple([1] * 36))
    ENCODER_CACHE_TTL_S: float = 0.01
    ENCODER_PID_KP: float = 2.0
    ENCODER_PID_KI: float = 0.0
    ENCODER_PID_KD: float = 0.03
    ENCODER_PID_TOL_DEG: float = 0.5
    ENCODER_PID_TIMEOUT_S: float = 0.15
    ENCODER_PID_DT_S: float = 0.02
    ENCODER_PWM_CORRECTION_LIMIT_US: float = 250.0
