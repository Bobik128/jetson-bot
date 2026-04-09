from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_teleop import DualsenseTeleopConfig


@TeleoperatorConfig.register_subclass("dummy_so_leader_plus_dualsense")
@dataclass
class DummySOLeaderPlusDualsenseConfig(TeleoperatorConfig):
    ds: DualsenseTeleopConfig = field(default_factory=DualsenseTeleopConfig)

    # Optional prefixes to prevent key collisions
    prefix_so: str = ""
    prefix_ds: str = ""

    # Fixed dummy SO101 arm action values
    # Adjust these to your robot's safe neutral pose.
    dummy_arm_action: dict[str, float] = field(
        default_factory=lambda: {
            "arm_shoulder_lift.pos": -75.0,
            "arm_elbow_flex.pos": 70.0,
            "arm_wrist_flex.pos": 20.0,
            "arm_gripper.pos": 0.0,
        }
    )