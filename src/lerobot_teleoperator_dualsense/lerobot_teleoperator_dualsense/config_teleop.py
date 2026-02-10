from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("dualsense_pygame")
@dataclass
class DualsenseTeleopConfig(TeleoperatorConfig):
    """
    Config for DualSense (pygame) teleoperator.

    You can override axis/button indices here without touching the code.
    """

    joystick_index: int = 0

    # output toggles
    use_gripper: bool = True
    use_home: bool = True

    # axes
    axis_forward: int = 1
    axis_turn: int = 0
    axis_turbo: int = 2

    invert_forward: bool = True
    invert_turn: bool = False

    deadzone: float = 0.08
    expo: float = 0.25

    # base limits
    max_v_mps: float = 0.35
    max_w_radps: float = 2.0

    # turbo
    use_turbo: bool = True
    turbo_min_scale: float = 0.35

    # buttons
    btn_gripper_open: int = 5   # R1
    btn_gripper_close: int = 4  # L1
    btn_home: int = 12          # PS

    # hat speed stepping
    use_hat_speed_steps: bool = True
    hat_index: int = 0
    speed_levels: Sequence[Tuple[float, float]] = field(
        default_factory=lambda: [(0.20, 1.0), (0.30, 2.0), (0.40, 3.0)]
    )

    # gripper semantic values (match your robot expectation)
    gripper_open_value: float = 100.0
    gripper_close_value: float = 0.0
