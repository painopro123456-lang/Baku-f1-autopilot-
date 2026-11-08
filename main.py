from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mss
import numpy as np
import pyautogui
import pygetwindow as gw
from pynput import keyboard as global_keyboard
from PIL import Image, ImageTk
from autopilot_core import AdaptiveController

APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "BakuAutopilot"
APP_DIR.mkdir(parents=True, exist_ok=True)
LEARNING_FILE = APP_DIR / "learning.json"
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class Learner:
    def __init__(self):
        self.data = {
            "laps": [], "events": [], "steer_gain": 0.78,
            "integral_gain": 0.025, "derivative_gain": 0.20,
            "throttle_limit": 0.82, "best_time": None
        }
        try:
            if LEARNING_FILE.exists():
                saved = json.loads(LEARNING_FILE.read_text(encoding="utf-8"))
                if isinstance(saved, dict):
                    self.data.update(saved)
        except (OSError, ValueError, TypeError):
            pass
        self.data.setdefault("laps", [])
        self.data.setdefault("events", [])

    def save(self):
        APP_DIR.mkdir(parents=True, exist_ok=True)
        temp = LEARNING_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temp.replace(LEARNING_FILE)

    def record(self, event, detail=""):
        self.data["events"].append({
            "time": time.time(), "event": event, "detail": str(detail)[:240]
        })
        self.data["events"] = self.data["events"][-500:]
        self.save()

    def lap(self, seconds):
        entry = {
            "time": round(seconds, 3),
            "timestamp": time.time(),
            "steer_gain": float(self.data.get("steer_gain", 0.78)),
            "throttle_limit": float(self.data.get("throttle_limit", 0.82)),
        }
        self.data["laps"].append(entry)
        self.data["laps"] = self.data["laps"][-100:]
        valid = [x["time"] for x in self.data["laps"] if x.get("time", 0) > 0]
        self.data["best_time"] = min(valid) if valid else None
        # Do not auto-tune from lap time alone: a faster lap is not proof
        # that a particular steering or throttle change caused the improvement.
        self.save()


class Vision:
    """Estimate a road corridor from several horizontal slices of the live frame."""
    def __init__(self):
        self.last_centers = None
        self.last_frame_gray = None
        self.last_time = time.monotonic()

    @staticmethod
    def _road_mask(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Broad asphalt cue. The connected-component and multi-slice checks below
        # reject much of the sky/HUD, but this still needs calibration per graphics setting.
        low_sat = cv2.inRange(hsv, np.array([0, 0, 32]), np.array([180, 105, 205]))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture = cv2.Canny(gray, 35, 105)
        mask = cv2.bitwise_and(low_sat, cv2.bitwise_not(texture))
        kernel = np.ones((7, 7), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def analyze(self, frame):
        h, w = frame.shape[:2]
        # Exclude outer HUD and the far horizon. Keep enough vertical depth for preview.
        x0, x1 = int(w * .08), int(w * .92)
        y0, y1 = int(h * .42), int(h * .97)
        roi = frame[y0:y1, x0:x1]
        mask = self._road_mask(roi)
        rh, rw = mask.shape
        centers, widths, scores, ys = [], [], [], []
        for fraction in (.38, .52, .66, .80, .94):
            y = min(rh - 1, int(rh * fraction))
            band = mask[max(0, y - 7):min(rh, y + 8), :]
            column_score = np.count_nonzero(band, axis=0).astype(np.float32)
            smooth = np.convolve(column_score, np.ones(19, dtype=np.float32), mode="same")
            # Find the strongest broad region, then require plausible road width.
            peak = int(np.argmax(smooth))
            threshold = max(2.0, float(smooth[peak]) * .22)
            left = peak
            right = peak
            while left > 0 and smooth[left - 1] >= threshold:
                left -= 1
            while right < rw - 1 and smooth[right + 1] >= threshold:
                right += 1
            width = right - left
            score = float(smooth[peak] / 15.0)
            if rw * .12 <= width <= rw * .94 and score >= 1.5:
                centers.append((left + right) / 2)
                widths.append(width)
                scores.append(min(1.0, score / 12.0))
                ys.append(fraction)
        if len(centers) < 3:
            return {"valid": False, "confidence": 0.0, "centers": [], "curvature": 0.0,
                    "offset": 0.0, "preview": mask}
        # Normalize offsets relative to each estimated corridor width.
        offsets = [(c - rw / 2) / max(1.0, width / 2) for c, width in zip(centers, widths)]
        # Far-to-near line trend: curvature is the change in lateral offset by depth.
        near = float(np.mean(offsets[-2:]))
        far = float(np.mean(offsets[:2]))
        curvature = float(np.clip((near - far) * 1.8, -1.0, 1.0))
        confidence = float(np.mean(scores)) * min(1.0, len(centers) / 5.0)
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        now = time.monotonic()
        flow = 0.0
        if self.last_frame_gray is not None:
            vectors = cv2.calcOpticalFlowFarneback(
                self.last_frame_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude = cv2.magnitude(vectors[..., 0], vectors[..., 1])
            # Relative pixel motion only; never presented as calibrated vehicle speed.
            flow = float(np.mean(magnitude)) / 12.0
        self.last_frame_gray = gray
        self.last_time = now
        return {
            "valid": True, "confidence": confidence, "centers": centers,
            "offset": float(np.clip(np.mean(offsets[-2:]), -1.0, 1.0)),
            "curvature": curvature, "flow": float(np.clip(flow, 0, 1)),
            "preview": mask, "roi_x": (x0, x1), "roi_y": (y0, y1), "sample_y": ys
        }


class Driver:
    def __init__(self, learner):
        self.learner = learner
        self.vision = Vision()
        self.controller = AdaptiveController()
        self.running = False
        self.enabled = False
        self.last_keys = set()
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.monotonic()
        self.confidence_misses = 0
        self.last_message = "Stopped — observe-only"
        self.target_title = "Roblox"
        self.last_output = None
        self.ready_frames = 0
        self.pit_exit_mode = False

    def game_is_focused(self):
        try:
            title = gw.getActiveWindowTitle() or ""
            return "roblox" in title.lower()
        except Exception:
            return False

    def release(self):
        for key in ("w", "a", "s", "d"):
            try:
                pyautogui.keyUp(key)
            except Exception:
                pass
        self.last_keys.clear()

    def set_keys(self, keys):
        if not self.game_is_focused():
            self.release()
            return False
        try:
            for key in self.last_keys - keys:
                pyautogui.keyUp(key)
            for key in keys - self.last_keys:
                pyautogui.keyDown(key)
            self.last_keys = set(keys)
            return True
        except Exception:
            self.release()
            return False

    def step(self, frame):
        result = self.vision.analyze(frame)
        settings = dict(self.learner.data)
        # Pit exit uses the game's own limiter; do not impose an artificial throttle cap.
        # The controller still reduces throttle / brakes when the detected road bends.
        if self.pit_exit_mode:
            settings["throttle_limit"] = 1.0
        output = self.controller.update(result, settings)
        self.last_output = output
        if output.mode.startswith("VISION LOST"):
            self.release()
            return f"Road unclear ({output.confidence:.2f}) — controls released", result

        if self.enabled:
            # Require several stable, confident frames before the first key press.
            if output.confidence < 0.35:
                self.ready_frames = 0
                self.release()
                return "Road confidence too low — waiting safely", result
            self.ready_frames += 1
            if self.ready_frames < 5:
                self.release()
                return f"Stabilizing road detection ({self.ready_frames}/5)", result

            # Fail closed: only send keys while the Roblox game is foreground.
            if not self.game_is_focused():
                self.release()
                return "Roblox not focused — keys released", result

            keys = set()
            # Pit limiter is enforced by the game; keep the normal controller output.
            steering = output.steering
            throttle = output.throttle
            if steering > 0.13:
                keys.add("d")
            elif steering < -0.13:
                keys.add("a")
            if output.brake >= 0.18:
                keys.add("s")
            elif throttle >= 0.18:
                keys.add("w")
            if not self.set_keys(keys):
                return "Roblox not focused — keys released", result
        else:
            keys = set()

        return (
            f"Road {output.confidence:.2f} | offset {result['offset']:+.2f} | "
            f"curve {result['curvature']:+.2f} | steer {output.steering:+.2f} | "
            f"relative-flow {self.controller.speed_proxy:.2f} | {output.mode}"
        ), result



class LapDetector:
    """Experimental visual checkpoint timer; arm while stationary at the start/finish line."""
    def __init__(self):
        self.reference = None
        self.started = None
        self.matches = 0
        self.minimum_lap_seconds = 25.0

    @staticmethod
    def signature(frame):
        h, w = frame.shape[:2]
        # Central scene crop avoids most HUD elements and uses a compact fingerprint.
        crop = frame[int(h * .22):int(h * .82), int(w * .20):int(w * .80)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)

    def arm(self, frame):
        self.reference = self.signature(frame)
        self.started = time.monotonic()
        self.matches = 0

    def update(self, frame):
        if self.reference is None or self.started is None:
            return None
        elapsed = time.monotonic() - self.started
        if elapsed < self.minimum_lap_seconds:
            return None
        current = self.signature(frame)
        difference = float(np.mean(cv2.absdiff(self.reference, current)))
        # Require consecutive similar frames to reduce one-frame false positives.
        self.matches = self.matches + 1 if difference < 10.0 else 0
        if self.matches >= 5:
            self.reference = current
            self.started = time.monotonic()
            self.matches = 0
            return elapsed
        return None


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Baku Autopilot")
        self.root.geometry("1000x720")
        self.root.minsize(820, 620)
        self.learner = Learner()
        self.driver = Driver(self.learner)
        self.stop_event = threading.Event()
        self.latest = None
        self.latest_result = None
        self.lap_detector = LapDetector()
        self.last_capture_frame = None
        self.status = tk.StringVar(value="Stopped — observe-only")
        self.metrics = tk.StringVar(value=self.metrics_text())
        self.live_telemetry = tk.StringVar(value="Waiting for live vision…")
        self.steering_gain = tk.DoubleVar(value=float(self.learner.data.get("steer_gain", 0.78)))
        self.throttle_limit = tk.DoubleVar(value=float(self.learner.data.get("throttle_limit", 0.82)))
        self.hotkey_listener = None
        self._build()
        self.root.bind("<F8>", lambda _e: self.stop())
        self.root.bind("<F9>", lambda _e: self.enable())
        self.root.bind("<F10>", lambda _e: self.switch_to_race())
        self._start_global_hotkeys()
        self.root.bind("<F6>", lambda _e: self.mark_lap())
        self.root.bind("<F7>", lambda _e: self.record_incident())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.refresh)

    def metrics_text(self):
        best = self.learner.data.get("best_time")
        laps = len(self.learner.data.get("laps", []))
        return f"Recorded laps: {laps}   Best recorded: {best if best else '—'}"

    def _build(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="BAKU AUTOPILOT", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, text="2006 Rouiller • brake bias 55 • WASD • no DRS / ERS").pack(anchor="w", pady=(2, 8))
        self.preview = ttk.Label(outer, text="Live game preview will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status, wraplength=940).pack(anchor="w", pady=8)
        ttk.Label(outer, textvariable=self.live_telemetry, wraplength=940).pack(anchor="w", pady=(0, 8))
        ttk.Label(outer, textvariable=self.metrics).pack(anchor="w", pady=(0, 8))
        settings = ttk.LabelFrame(outer, text="Driving settings", padding=8)
        settings.pack(fill="x", pady=(0, 8))
        ttk.Label(settings, text="Steering response").grid(row=0, column=0, sticky="w")
        ttk.Scale(settings, from_=0.40, to=1.15, variable=self.steering_gain,
                  command=self.update_settings).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(settings, text="Throttle limit").grid(row=1, column=0, sticky="w")
        ttk.Scale(settings, from_=0.25, to=1.0, variable=self.throttle_limit,
                  command=self.update_settings).grid(row=1, column=1, sticky="ew", padx=8)
        settings.columnconfigure(1, weight=1)
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Button(controls, text="Start observing", command=self.start_observing).pack(side="left", padx=(0, 7))
        ttk.Button(controls, text="Pit exit mode", command=self.begin_pit_exit).pack(side="left", padx=7)
        ttk.Button(controls, text="Enable race mode", command=self.begin_race_mode).pack(side="left", padx=7)
        ttk.Button(controls, text="Mark lap (F6)", command=self.mark_lap).pack(side="left", padx=7)
        ttk.Button(controls, text="Record incident (F7)", command=self.record_incident).pack(side="left", padx=7)
        ttk.Button(controls, text="STOP (F8)", command=self.stop).pack(side="right")
        ttk.Label(outer, text="Pit exit route: drive forward from garage → sharp left → pit lane → sharp left at pit-lane end. F10 switches to race mode once on track.").pack(anchor="w", pady=(10, 0))
        ttk.Label(outer, text="F6: lap checkpoint • F8: emergency stop • F9: toggle autopilot • F10: switch pit exit to race mode").pack(anchor="w", pady=(4, 0))

    def _start_global_hotkeys(self):
        # Global F8 works even while Roblox has focus; F9 toggles driving.
        def on_press(key):
            try:
                if key == global_keyboard.Key.f8:
                    self.root.after(0, self.stop)
                elif key == global_keyboard.Key.f9:
                    self.root.after(0, self.enable)
                elif key == global_keyboard.Key.f10:
                    self.root.after(0, self.switch_to_race)
            except RuntimeError:
                return False
        try:
            self.hotkey_listener = global_keyboard.Listener(on_press=on_press)
            self.hotkey_listener.daemon = True
            self.hotkey_listener.start()
        except Exception as exc:
            self.status.set(f"Global hotkeys unavailable ({exc}); keep the app accessible")

    def update_settings(self, _value=None):
        self.learner.data["steer_gain"] = round(float(self.steering_gain.get()), 3)
        self.learner.data["throttle_limit"] = round(float(self.throttle_limit.get()), 3)
        try:
            self.learner.save()
        except OSError as exc:
            self.status.set(f"Could not save settings: {exc}")

    def start_observing(self):
        if self.driver.running:
            return
        self.stop_event.clear()
        self.driver.running = True
        self.driver.enabled = False
        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.status.set("Observing only — verify the preview and road detection before enabling")

    def begin_pit_exit(self):
        if not self.driver.running:
            self.start_observing()
        self.driver.pit_exit_mode = True
        self.driver.enabled = True
        self.driver.ready_frames = 0
        self.status.set("GARAGE → PIT EXIT: follow the road through both left turns; game pit limiter handles pit speed; F10 once on track")
        self.root.after(900, self.root.iconify)

    def switch_to_race(self):
        if not self.driver.enabled:
            self.status.set("Autopilot is not active — start pit exit or race mode first")
            return
        self.driver.pit_exit_mode = False
        self.driver.ready_frames = 0
        self.driver.release()
        self.status.set("RACE MODE — normal adaptive controls; stabilizing vision")

    def begin_race_mode(self):
        # This button is an explicit command, never a toggle: clicking it while
        # pit-exit mode is active must switch modes, not accidentally disable driving.
        if not self.driver.running:
            self.start_observing()
        self.driver.enabled = True
        self.driver.pit_exit_mode = False
        self.driver.ready_frames = 0
        self.driver.release()
        self.status.set("RACE MODE ARMED — returning focus to Roblox; waiting for confident road detection")
        self.root.after(900, self.root.iconify)

    def enable(self):
        if not self.driver.running:
            self.start_observing()
        if self.driver.enabled:
            self.driver.enabled = False
            self.driver.ready_frames = 0
            self.driver.release()
            self.status.set("Autopilot disabled — observing only")
            return
        self.driver.enabled = True
        self.driver.pit_exit_mode = False
        self.driver.ready_frames = 0
        self.status.set("Autopilot armed — switching to Roblox; controls only engage when Roblox is foreground")
        # Minimize the dashboard so Roblox can receive focus. The capture loop
        # continues; it sends no keys unless the Roblox window is active.
        self.root.after(900, self.root.iconify)

    def mark_lap(self):
        if self.latest is None:
            self.status.set("Start observing first, then press F6 at the start/finish line")
            return
        self.lap_detector.arm(self.latest.copy())
        self.status.set("Visual lap checkpoint armed — drive across the line; timing is experimental")

    def record_incident(self):
        detail = self.status.get()
        self.learner.record("user_reported_incident", detail)
        self.status.set("Incident saved to learning history")

    def stop(self):
        self.driver.enabled = False
        self.driver.running = False
        self.driver.ready_frames = 0
        self.driver.pit_exit_mode = False
        self.stop_event.set()
        self.driver.release()
        self.driver.controller.reset()
        self.status.set("Stopped — all keys released")

    def capture_loop(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while not self.stop_event.is_set():
                try:
                    shot = np.asarray(sct.grab(monitor))
                    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                    # Analyze at a bounded resolution to reduce CPU use.
                    small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                    message, result = self.driver.step(small) if self.driver.running else ("Stopped", None)
                    display = small.copy()
                    if result and result.get("valid"):
                        x0, _x1 = result["roi_x"]
                        y0, _y1 = result["roi_y"]
                        points = [
                            (int(x0 + cx), int(y0 + frac * (_y1 - y0)))
                            for cx, frac in zip(result["centers"], result["sample_y"])
                        ]
                        if len(points) >= 2:
                            cv2.polylines(display, [np.asarray(points, dtype=np.int32)],
                                          False, (40, 220, 90), 3)
                            for point in points:
                                cv2.circle(display, point, 5, (30, 80, 255), -1)
                    self.latest = cv2.resize(display, (800, 450), interpolation=cv2.INTER_AREA)
                    self.latest_result = result
                    if result and result.get("valid"):
                        out = self.driver.last_output
                        telemetry = (f"Vision {result['confidence']:.0%}  |  "
                                     f"Track offset {result['offset']:+.2f}  |  "
                                     f"Curve {result['curvature']:+.2f}")
                        if out:
                            telemetry += (f"  |  Steering {out.steering:+.2f}  |  "
                                          f"Throttle {out.throttle:.0%}  |  Brake {out.brake:.0%}")
                        self.root.after(0, lambda m=telemetry: self.live_telemetry.set(m))
                    else:
                        self.root.after(0, lambda: self.live_telemetry.set("Road not confidently detected — inputs released"))
                    lap_seconds = self.lap_detector.update(small)
                    if lap_seconds is not None:
                        self.learner.lap(lap_seconds)
                        summary = f"Visual lap detected: {lap_seconds:.2f}s"
                        self.root.after(0, lambda: self.metrics.set(self.metrics_text()))
                        self.root.after(0, lambda m=summary: self.status.set(m))
                    else:
                        self.root.after(0, lambda m=message: self.status.set(m))
                except Exception as exc:
                    self.root.after(0, lambda e=str(exc): self.status.set(f"Capture error: {e}"))
                    self.driver.enabled = False
                    self.driver.running = False
                    self.driver.ready_frames = 0
                    self.stop_event.set()
                    self.driver.release()
                    break
                time.sleep(.07)
        self.driver.release()

    def refresh(self):
        if self.latest is not None:
            rgb = cv2.cvtColor(self.latest, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
        self.root.after(100, self.refresh)

    def close(self):
        self.stop()
        if self.hotkey_listener is not None:
            try:
                self.hotkey_listener.stop()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    App(tk.Tk()).root.mainloop()
