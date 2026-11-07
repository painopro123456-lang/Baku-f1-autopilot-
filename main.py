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
from PIL import Image, ImageTk

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
        entry = {"time": round(seconds, 3), "timestamp": time.time()}
        self.data["laps"].append(entry)
        self.data["laps"] = self.data["laps"][-100:]
        valid = [x["time"] for x in self.data["laps"] if x.get("time", 0) > 0]
        self.data["best_time"] = min(valid) if valid else None
        # Small, bounded adjustments only. Actual driving quality still needs testing.
        if len(valid) >= 2 and seconds < valid[-2]:
            self.data["steer_gain"] = min(1.15, float(self.data["steer_gain"]) * 1.01)
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
            flow = float(np.mean(cv2.absdiff(gray, self.last_frame_gray))) / 35.0
        self.last_frame_gray = gray
        self.last_time = now
        return {
            "valid": True, "confidence": confidence, "centers": centers,
            "offset": float(np.clip(np.mean(offsets[-2:]), -1.0, 1.0)),
            "curvature": curvature, "flow": float(np.clip(flow, 0, 1)),
            "preview": mask, "roi_x": (x0, x1), "roi_y": (y0, y1)
        }


class Driver:
    def __init__(self, learner):
        self.learner = learner
        self.vision = Vision()
        self.running = False
        self.enabled = False
        self.last_keys = set()
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.monotonic()
        self.confidence_misses = 0
        self.last_message = "Stopped — observe-only"
        self.target_title = "Roblox"

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
        for key in self.last_keys - keys:
            pyautogui.keyUp(key)
        for key in keys - self.last_keys:
            pyautogui.keyDown(key)
        self.last_keys = set(keys)
        return True

    def step(self, frame):
        result = self.vision.analyze(frame)
        if not result["valid"] or result["confidence"] < .10:
            self.confidence_misses += 1
            self.release()
            return "Road unclear — keys released", result
        self.confidence_misses = 0
        error = result["offset"]
        curve = result["curvature"]
        now = time.monotonic()
        dt = max(.02, min(.20, now - self.previous_time))
        self.previous_time = now
        self.integral = float(np.clip(self.integral + error * dt, -1.2, 1.2))
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        d = self.learner.data
        steer = (float(d["steer_gain"]) * error
                 + float(d["integral_gain"]) * self.integral
                 + float(d["derivative_gain"]) * derivative * .04
                 + .30 * curve)
        steer = float(np.clip(steer, -1.0, 1.0))
        severity = min(1.0, abs(curve) * .75 + abs(error) * .55)
        throttle = float(d.get("throttle_limit", .82)) * (1.0 - .62 * severity)
        keys = set()
        if self.enabled:
            # Fail closed: no input unless Roblox is the foreground window.
            if not self.game_is_focused():
                self.release()
                return "Roblox not focused — keys released", result
            if steer > .12:
                keys.add("d")
            elif steer < -.12:
                keys.add("a")
            if severity > .72:
                keys.add("s")
            elif throttle > .15:
                keys.add("w")
            if not self.set_keys(keys):
                return "Roblox not focused — keys released", result
        return (f"Road {result['confidence']:.2f} | offset {error:+.2f} | "
                f"curve {curve:+.2f} | steering {steer:+.2f} | "
                f"{'BRAKE' if 's' in keys else 'throttle' if 'w' in keys else 'coast'}"), result


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
        self.lap_started = None
        self.status = tk.StringVar(value="Stopped — observe-only")
        self.metrics = tk.StringVar(value=self.metrics_text())
        self._build()
        self.root.bind("<F8>", lambda _e: self.stop())
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
        ttk.Label(outer, textvariable=self.metrics).pack(anchor="w", pady=(0, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Button(controls, text="Start observing", command=self.start_observing).pack(side="left", padx=(0, 7))
        ttk.Button(controls, text="Enable autopilot", command=self.enable).pack(side="left", padx=7)
        ttk.Button(controls, text="Mark lap (F6)", command=self.mark_lap).pack(side="left", padx=7)
        ttk.Button(controls, text="Record incident (F7)", command=self.record_incident).pack(side="left", padx=7)
        ttk.Button(controls, text="STOP (F8)", command=self.stop).pack(side="right")
        ttk.Label(outer, text="Safety: keep Roblox visible. Control is active only while a Roblox window is focused. F8 releases all keys.").pack(anchor="w", pady=(10, 0))

    def start_observing(self):
        if self.driver.running:
            return
        self.stop_event.clear()
        self.driver.running = True
        self.driver.enabled = False
        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.status.set("Observing only — check that the preview shows the track")

    def enable(self):
        if not self.driver.running:
            self.start_observing()
        if not self.driver.game_is_focused():
            messagebox.showwarning("Roblox not focused", "Click the Roblox game window first, then enable autopilot.")
            return
        self.driver.enabled = True
        self.status.set("Autopilot enabled — F8 stops immediately")

    def mark_lap(self):
        now = time.monotonic()
        if self.lap_started is None:
            self.lap_started = now
            self.status.set("Lap timer started — press F6 again at the finish line")
            return
        seconds = now - self.lap_started
        self.lap_started = now
        self.learner.lap(seconds)
        self.metrics.set(self.metrics_text())
        self.status.set(f"Lap recorded: {seconds:.3f}s — best: {self.learner.data.get('best_time')}s")

    def record_incident(self):
        detail = self.status.get()
        self.learner.record("user_reported_incident", detail)
        self.status.set("Incident saved to learning history")

    def stop(self):
        self.driver.enabled = False
        self.driver.running = False
        self.stop_event.set()
        self.driver.release()
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
                    self.latest = cv2.resize(small, (800, 450), interpolation=cv2.INTER_AREA)
                    self.latest_result = result
                    self.root.after(0, lambda m=message: self.status.set(m))
                except Exception as exc:
                    self.root.after(0, lambda e=str(exc): self.status.set(f"Capture error: {e}"))
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
        self.root.destroy()


if __name__ == "__main__":
    App(tk.Tk()).root.mainloop()
