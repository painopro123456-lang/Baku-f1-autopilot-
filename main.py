from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image, ImageTk

APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / "BakuAutopilot"
APP_DIR.mkdir(parents=True, exist_ok=True)
LEARNING_FILE = APP_DIR / "learning.json"

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

class Learner:
    def __init__(self):
        self.data = {"laps": 0, "best_time": None, "events": [], "steer_gain": 0.34, "throttle_limit": 0.55}
        try:
            if LEARNING_FILE.exists():
                self.data.update(json.loads(LEARNING_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass

    def record(self, event, detail=""):
        self.data["events"].append({"time": time.time(), "event": event, "detail": detail})
        self.data["events"] = self.data["events"][-500:]
        LEARNING_FILE.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

class Driver:
    """Conservative image heuristic; requires real-game calibration before fast driving."""
    def __init__(self, learner):
        self.learner = learner
        self.running = False
        self.enabled = False
        self.last_keys = set()
        self.last_center_error = 0.0
        self.last_change = time.time()

    def release(self):
        for key in ("w", "a", "s", "d"):
            try: pyautogui.keyUp(key)
            except Exception: pass
        self.last_keys.clear()

    def set_keys(self, keys):
        for key in self.last_keys - keys: pyautogui.keyUp(key)
        for key in keys - self.last_keys: pyautogui.keyDown(key)
        self.last_keys = set(keys)

    @staticmethod
    def road_error(frame):
        h, w = frame.shape[:2]
        # Focus on the lower-middle road area; reject the HUD and distant horizon.
        roi = frame[int(h * 0.48):int(h * 0.91), int(w * 0.12):int(w * 0.88)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Asphalt is often low-saturation and mid-dark. This is a generic cue,
        # not a guaranteed road classifier; walls/shadows can look similar.
        mask = cv2.inRange(hsv, np.array([0, 0, 35]), np.array([180, 95, 190]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        rows, cols = np.where(mask > 0)
        if len(cols) < 250:
            return None, 0.0
        # Use the lower half of the ROI, where the drivable surface should be widest.
        y_cut = roi.shape[0] * 0.55
        near = cols[rows >= y_cut]
        if len(near) < 100:
            near = cols
        center = float(np.median(near))
        error = (center - roi.shape[1] / 2) / max(1, roi.shape[1] / 2)
        confidence = min(1.0, len(near) / 3000.0)
        return float(np.clip(error, -1, 1)), confidence

    def step(self, frame):
        error, confidence = self.road_error(frame)
        if error is None or confidence < 0.08:
            self.release()
            return "Road not confidently detected — controls released"
        # Low-gain centering only. Avoid aggressive inputs until validated.
        steer = error * float(self.learner.data.get("steer_gain", 0.34))
        keys = set()
        if self.enabled:
            keys.add("w")
            if steer > 0.10: keys.add("d")
            elif steer < -0.10: keys.add("a")
            # No automatic reverse or hard braking in this uncalibrated prototype.
            self.set_keys(keys)
        return f"Road cue {confidence:.2f} | offset {error:+.2f} | steer {steer:+.2f}"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Baku Autopilot")
        self.root.geometry("900x650")
        self.root.minsize(760, 560)
        self.learner = Learner()
        self.driver = Driver(self.learner)
        self.stop_event = threading.Event()
        self.latest = None
        self.status = tk.StringVar(value="Stopped — observe-only")
        self.info = tk.StringVar(value="2006 Rouiller • brake bias 55 • WASD • no DRS / ERS")
        self._build()
        self.root.bind("<F8>", lambda _e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.refresh)

    def _build(self):
        outer = ttk.Frame(self.root, padding=14); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="BAKU AUTOPILOT", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(outer, textvariable=self.info).pack(anchor="w", pady=(2, 10))
        self.preview = ttk.Label(outer, text="Live game preview will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=8)
        controls = ttk.Frame(outer); controls.pack(fill="x")
        ttk.Button(controls, text="Start observing", command=self.start_observing).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Enable keyboard control", command=self.enable).pack(side="left", padx=8)
        ttk.Button(controls, text="STOP (F8)", command=self.stop).pack(side="right")
        ttk.Label(outer, text="Safety: keep Roblox visible. F8 or Stop releases all keys. Start in observe-only mode.").pack(anchor="w", pady=(10, 0))

    def start_observing(self):
        if self.driver.running: return
        self.stop_event.clear()
        self.driver.running = True
        self.driver.enabled = False
        threading.Thread(target=self.capture_loop, daemon=True).start()
        self.status.set("Observing only — inspect preview before enabling controls")

    def enable(self):
        if not self.driver.running:
            self.start_observing()
        self.driver.enabled = True
        self.status.set("Keyboard control enabled — press F8 to stop")

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
                    shot = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                    self.latest = cv2.resize(frame, (800, 450))
                    message = self.driver.step(frame) if self.driver.running else "Stopped"
                    self.root.after(0, lambda m=message: self.status.set(m))
                except Exception as exc:
                    self.root.after(0, lambda e=str(exc): self.status.set(f"Capture error: {e}"))
                    self.driver.release()
                    break
                time.sleep(0.06)
        self.driver.release()

    def refresh(self):
        if self.latest is not None:
            rgb = cv2.cvtColor(self.latest, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
        self.root.after(100, self.refresh)

    def close(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    App(tk.Tk()).root.mainloop()
