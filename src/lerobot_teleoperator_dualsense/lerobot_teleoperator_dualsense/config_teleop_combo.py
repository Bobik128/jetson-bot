from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from lerobot.teleoperators.config import TeleoperatorConfig

from .config_teleop import DualsenseTeleopConfig
from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig


@TeleoperatorConfig.register_subclass("so_leader_plus_dualsense")
@dataclass
class SOLeaderPlusDualsenseConfig(TeleoperatorConfig):
    # Sub-configs
    so: SO101LeaderConfig = field(default_factory=lambda: SO101LeaderConfig(port="/dev/ttyACM0", id="the_leader"))
    ds: DualsenseTeleopConfig = field(default_factory=DualsenseTeleopConfig)

    # Behavior
    allow_partial: bool = True

    # Optional prefixes to prevent key collisions
    prefix_so: str = ""
    prefix_ds: str = ""