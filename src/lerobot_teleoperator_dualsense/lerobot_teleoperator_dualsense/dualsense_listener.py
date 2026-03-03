from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pygame


@dataclass
class DualsenseEpisodeButtons:
    """
    Configure which controller buttons trigger which record-loop events.

    Button indices are the pygame joystick button numbers (js.get_button(idx)).

    Defaults are common on Linux/SDL but NOT guaranteed across OS/drivers.
    """
    btn_pause_toggle: int = 9        # Options
    btn_exit_early: int = 1          # Circle
    btn_rerecord_episode: int = 2    # Square
    btn_stop_recording: int = 8      # Create/Share

    # Debounce / polling
    poll_hz: float = 60.0
    debounce_sec: float = 0.20

    # Which key in events dict represents pause state
    pause_key: str = "paused"


class DualsenseEpisodeListener:
    """
    Background poller that watches a pygame joystick for button presses and
    mutates the shared `events` dict used by lerobot.record_loop().

    IMPORTANT:
      - This thread does NOT call pygame.event.pump()/get().
      - Your main teleop loop already calls pygame.event.pump() at FPS, which keeps
        joystick state fresh.
    """

    def __init__(
        self,
        js: pygame.joystick.Joystick,
        events: Dict[str, Any],
        mapping: DualsenseEpisodeButtons,
    ):
        self._js = js
        self._events = events
        self._m = mapping

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last = {}  # btn_idx -> bool
        self._last_fire = {}  # action_name -> timestamp

        # Ensure keys exist
        self._events.setdefault("stop_recording", False)
        self._events.setdefault("rerecord_episode", False)
        self._events.setdefault("exit_early", False)
        self._events.setdefault(self._m.pause_key, False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

    # -------- internals --------

    def _debounced(self, action: str) -> bool:
        now = time.monotonic()
        last = self._last_fire.get(action, 0.0)
        if (now - last) < self._m.debounce_sec:
            return False
        self._last_fire[action] = now
        return True

    def _pressed_edge(self, btn: int) -> bool:
        # Rising edge detection: False -> True
        cur = False
        try:
            if btn >= 0 and btn < self._js.get_numbuttons():
                cur = bool(self._js.get_button(btn))
        except Exception:
            cur = False

        prev = self._last.get(btn, False)
        self._last[btn] = cur
        val = (not prev) and cur
        if val:
            print("bressed button", btn)
            print(self._events)
        return val

    def _run(self) -> None:
        dt = 1.0 / max(1.0, float(self._m.poll_hz))
        while not self._stop.is_set():
            # Pause toggle
            if self._pressed_edge(self._m.btn_pause_toggle) and self._debounced("pause"):
                self._events[self._m.pause_key] = not bool(self._events.get(self._m.pause_key, False))

            # Exit episode early
            if self._pressed_edge(self._m.btn_exit_early) and self._debounced("exit_early"):
                self._events["exit_early"] = True

            # Re-record episode
            if self._pressed_edge(self._m.btn_rerecord_episode) and self._debounced("rerecord_episode"):
                self._events["rerecord_episode"] = True

            # Stop recording entirely
            if self._pressed_edge(self._m.btn_stop_recording) and self._debounced("stop_recording"):
                self._events["stop_recording"] = True

            time.sleep(dt)