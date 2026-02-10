from __future__ import annotations

from typing import Any, Dict

from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.teleoperators.so_leader.so_leader import SO101Leader
from .dualsense_teleop import DualsenseTeleop
from .config_teleop_combo import SOLeaderPlusDualsenseConfig


class SOLeaderPlusDualsense(Teleoperator):
    """
    Combined teleoperator:
      - SOLeader provides arm joint positions
      - DualsenseTeleop provides base velocity commands

    Output is a single action dict.
    """

    config_class = SOLeaderPlusDualsenseConfig
    name = "so_leader_plus_dualsense"

    def __init__(self, config: SOLeaderPlusDualsenseConfig):
        super().__init__(config)
        self.config = config

        self.so = SO101Leader(config.so)
        self.ds = DualsenseTeleop(config.ds)

    @property
    def action_features(self) -> dict[str, type]:
        feats: dict[str, type] = {}
        # merge features (with optional prefixes)
        for k, t in self.so.action_features.items():
            feats[f"{self.config.prefix_so}{k}"] = t
        for k, t in self.ds.action_features.items():
            feats[f"{self.config.prefix_ds}{k}"] = t

        # Sanity: ensure no collisions AFTER prefixing
        if len(feats) != (len(self.so.action_features) + len(self.ds.action_features)):
            raise ValueError(
                "Action feature key collision between SOLeader and DualSense. "
                "Set prefix_so/prefix_ds to avoid collisions."
            )
        return feats

    @property
    def feedback_features(self) -> dict[str, type]:
        # neither provides feedback currently
        return {}

    @property
    def is_connected(self) -> bool:
        return bool(self.so.is_connected and self.ds.is_connected)

    @property
    def is_calibrated(self) -> bool:
        # SOLeader calibration is real; DualSense always "calibrated"
        return bool(self.so.is_calibrated and self.ds.is_calibrated)

    def connect(self, calibrate: bool = True) -> None:
        # Connect both. If one fails and allow_partial is False, raise.
        so_ok = ds_ok = False

        try:
            self.so.connect(calibrate=calibrate)
            so_ok = True
        except Exception:
            if not self.config.allow_partial:
                raise

        try:
            self.ds.connect(calibrate=calibrate)
            ds_ok = True
        except Exception:
            if not self.config.allow_partial:
                raise

        if not (so_ok or ds_ok):
            raise RuntimeError("Failed to connect both SOLeader and DualSense.")

    def calibrate(self) -> None:
        # Delegate. SOLeader may prompt calibration. DualSense does nothing.
        self.so.calibrate()
        self.ds.calibrate()

    def configure(self) -> None:
        # Configure SO; DualSense doesn't need configuration
        self.so.configure()
        self.ds.configure()

    def get_action(self) -> dict[str, Any]:
        action: Dict[str, Any] = {}

        so_action: Dict[str, Any] = {}
        ds_action: Dict[str, Any] = {}

        # Read both. If one fails and allow_partial True, keep going.
        try:
            if self.so.is_connected:
                so_action = self.so.get_action()
        except Exception:
            if not self.config.allow_partial:
                raise

        try:
            if self.ds.is_connected:
                ds_action = self.ds.get_action()
        except Exception:
            if not self.config.allow_partial:
                raise

        # Apply prefixes
        for k, v in so_action.items():
            action[f"{self.config.prefix_so}{k}"] = v
        for k, v in ds_action.items():
            action[f"{self.config.prefix_ds}{k}"] = v

        # Collision check (runtime, just in case)
        if len(action) != (len(so_action) + len(ds_action)):
            raise ValueError(
                "Runtime action key collision between SOLeader and DualSense. "
                "Set prefix_so/prefix_ds to avoid collisions."
            )

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # optional: route feedback to ds/so. SOLeader raises NotImplemented.
        # We'll ignore by default.
        return

    def disconnect(self) -> None:
        # Disconnect both (best-effort)
        try:
            if self.ds.is_connected:
                self.ds.disconnect()
        finally:
            if self.so.is_connected:
                self.so.disconnect()
