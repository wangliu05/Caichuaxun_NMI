from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class VisualState:
    frame_idx: int
    target_type: str
    position_2d: Optional[np.ndarray] = None
    tangent_2d: Optional[np.ndarray] = None
    centerline_2d: Optional[np.ndarray] = None
    position_3d: Optional[np.ndarray] = None
    tangent_3d: Optional[np.ndarray] = None
    desired_tangent_3d: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class MagneticObjective:
    kind: str
    position_m: np.ndarray
    desired_field_t: Optional[np.ndarray] = None
    desired_gradient_t_per_m: Optional[np.ndarray] = None
    desired_gradient_drive_n: Optional[np.ndarray] = None
    metadata: dict = None
