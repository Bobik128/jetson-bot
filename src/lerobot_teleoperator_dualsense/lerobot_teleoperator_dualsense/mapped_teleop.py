from lerobot.teleoperators import Teleoperator

MAP = {
    "shoulder_lift.pos": "arm_shoulder_lift.pos",
    "elbow_flex.pos":    "arm_elbow_flex.pos",
    "wrist_flex.pos":    "arm_wrist_flex.pos",
    "gripper.pos":       "arm_gripper.pos",
    "motor_linear.vel":  "motor_linear.vel",
    "motor_angular.vel": "motor_angular.vel"
}

class MappedTeleop(Teleoperator):
    def __init__(self, inner: Teleoperator, map_: dict[str, str] = MAP):
        self._inner = inner
        self._map = dict(map_)

    # ---- required API (forward to inner) ----
    @property
    def action_features(self):
        # IMPORTANT: features must match remapped keys
        # We'll remap the inner action_features "names" if present.
        feats = self._inner.action_features
        remapped = {}
        for k, v in feats.items():
            if isinstance(v, dict) and "names" in v:
                names = [self._map.get(n, n) for n in v["names"]]
                vv = dict(v)
                vv["names"] = names
                remapped[k] = vv
            else:
                remapped[k] = v
        return remapped

    @property
    def feedback_features(self):
        return self._inner.feedback_features

    @property
    def is_connected(self):
        return self._inner.is_connected

    @property
    def is_calibrated(self):
        return getattr(self._inner, "is_calibrated", True)

    def connect(self):
        return self._inner.connect()

    def disconnect(self):
        return self._inner.disconnect()

    def configure(self, **kwargs):
        # some teleops expose configure; forward if available
        if hasattr(self._inner, "configure"):
            return self._inner.configure(**kwargs)
        return None

    def calibrate(self):
        if hasattr(self._inner, "calibrate"):
            return self._inner.calibrate()
        return None

    def send_feedback(self, feedback):
        if hasattr(self._inner, "send_feedback"):
            return self._inner.send_feedback(feedback)
        return None

    # ---- the actual remap ----
    def get_action(self):
        raw = self._inner.get_action()
        # keep keys not in map as-is, remap those that are in map
        out = {}
        for k, v in raw.items():
            out[self._map.get(k, k)] = v
        return out