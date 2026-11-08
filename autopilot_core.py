from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass
class ControlOutput:
    steering: float
    throttle: float
    brake: float
    severity: float
    mode: str
    confidence: float


class AdaptiveController:
    """
    Conservative screen-vision controller for keyboard-only driving.
    It does not assume access to game telemetry or car physics.
    """
    def __init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.monotonic()
        self.filtered_steer = 0.0
        self.filtered_curve = 0.0
        self.last_good_time = 0.0
        self.confidence_failures = 0
        self.speed_proxy = 0.0
        self.corner_memory = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.monotonic()
        self.filtered_steer = 0.0
        self.filtered_curve = 0.0
        self.confidence_failures = 0
        self.speed_proxy = 0.0
        self.corner_memory = 0.0

    def update(self, vision: dict, settings: dict) -> ControlOutput:
        now = time.monotonic()
        dt = max(0.025, min(0.15, now - self.previous_time))
        self.previous_time = now

        confidence = float(vision.get("confidence", 0.0))
        if not vision.get("valid") or confidence < 0.18:
            self.confidence_failures += 1
            self.integral *= 0.65
            self.filtered_steer *= 0.55
            return ControlOutput(0.0, 0.0, 0.0, 1.0, "VISION LOST — RELEASE", confidence)

        self.confidence_failures = 0
        self.last_good_time = now
        error = max(-1.0, min(1.0, float(vision.get("offset", 0.0))))
        curve = max(-1.0, min(1.0, float(vision.get("curvature", 0.0))))
        flow = max(0.0, min(1.0, float(vision.get("flow", 0.0))))

        # Flow is only a relative motion indicator, NOT a calibrated vehicle speed.
        self.speed_proxy = 0.78 * self.speed_proxy + 0.22 * flow
        self.filtered_curve = 0.62 * self.filtered_curve + 0.38 * curve
        curve = self.filtered_curve

        kp = float(settings.get("steer_gain", 0.78))
        ki = float(settings.get("integral_gain", 0.025))
        kd = float(settings.get("derivative_gain", 0.20))
        self.integral = max(-0.8, min(0.8, self.integral + error * dt))
        derivative = max(-3.0, min(3.0, (error - self.previous_error) / dt))
        self.previous_error = error

        # Look-ahead curve feed-forward plus bounded PID correction.
        raw_steer = kp * error + ki * self.integral + kd * derivative * 0.035 + 0.34 * curve
        raw_steer = max(-1.0, min(1.0, raw_steer))
        max_change = 2.8 * dt
        self.filtered_steer += max(-max_change, min(max_change, raw_steer - self.filtered_steer))
        steering = max(-1.0, min(1.0, self.filtered_steer))

        lateral = abs(error)
        bend = abs(curve)
        # Brake earlier for a strong bend, especially when relative image motion is high.
        severity = max(0.0, min(1.0, 0.52 * bend + 0.34 * lateral + 0.14 * self.speed_proxy))
        throttle_limit = max(0.25, min(1.0, float(settings.get("throttle_limit", 0.82))))
        brake = 0.0
        throttle = throttle_limit

        if bend > 0.58 or (bend > 0.43 and self.speed_proxy > 0.32):
            brake = min(1.0, 0.30 + 0.85 * bend + 0.30 * self.speed_proxy)
            throttle = 0.0
            mode = "BRAKE — CORNER"
        elif lateral > 0.72:
            brake = 0.28 + 0.35 * lateral
            throttle = 0.0
            mode = "BRAKE — RECOVER LINE"
        else:
            reduction = min(0.72, 0.60 * bend + 0.45 * lateral)
            throttle = max(0.22, throttle_limit * (1.0 - reduction))
            mode = "THROTTLE" if throttle > 0.35 else "COAST / LIGHT THROTTLE"

        # Never request throttle and brake together.
        if brake > 0:
            throttle = 0.0
        return ControlOutput(steering, throttle, brake, severity, mode, confidence)
