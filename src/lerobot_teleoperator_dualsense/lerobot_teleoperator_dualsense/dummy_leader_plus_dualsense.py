from __future__ import annotations

from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator

from .dualsense_teleop import DualsenseTeleop
from .config_dummy_teleop_combo import DummySOLeaderPlusDualsenseConfig


class DummySOLeaderPlusDualsense(Teleoperator):
    """
    Teleoperator that uses:
      - fixed dummy SO101 arm joint values
      - real DualSense base velocity commands

    Useful when the physical SO101 leader arm is unavailable.
    """

    config_class = DummySOLeaderPlusDualsenseConfig
    name = "dummy_so_leader_plus_dualsense"

    def __init__(self, config: DummySOLeaderPlusDualsenseConfig):
        super().__init__(config)
        self.config = config
        self.ds = DualsenseTeleop(config.ds)

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}

        for key in self.config.dummy_arm_action.keys():
            feats[f"{self.config.prefix_so}{key}"] = float

        for k, t in self.ds.action_features.items():
            feats[f"{self.config.prefix_ds}{k}"] = t

        return feats

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return bool(self.ds.is_connected)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.ds.is_calibrated)

    def connect(self, calibrate: bool = True) -> None:
        self.ds.connect(calibrate=calibrate)

    def calibrate(self) -> None:
        self.ds.calibrate()

    def configure(self) -> None:
        self.ds.configure()

    def get_action(self) -> dict[str, Any]:
        action: dict[str, Any] = {}

        for k, v in self.config.dummy_arm_action.items():
            action[f"{self.config.prefix_so}{k}"] = float(v)

        ds_action: dict[str, Any] = {}
        if self.ds.is_connected:
            ds_action = self.ds.get_action()

        for k, v in ds_action.items():
            action[f"{self.config.prefix_ds}{k}"] = v

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        return

    def disconnect(self) -> None:
        if self.ds.is_connected:
            self.ds.disconnect()