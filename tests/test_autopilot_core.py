import unittest

from autopilot_core import AdaptiveController


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = AdaptiveController()
        self.settings = {
            "steer_gain": 0.78,
            "integral_gain": 0.025,
            "derivative_gain": 0.20,
            "throttle_limit": 0.82,
        }

    def test_lost_vision_requests_no_controls(self):
        out = self.controller.update({"valid": False, "confidence": 0.0}, self.settings)
        self.assertEqual(out.throttle, 0.0)
        self.assertEqual(out.brake, 0.0)
        self.assertEqual(out.steering, 0.0)
        self.assertIn("VISION LOST", out.mode)

    def test_controls_are_bounded_and_never_brake_and_throttle(self):
        for _ in range(20):
            out = self.controller.update({
                "valid": True, "confidence": 0.9, "offset": 0.3,
                "curvature": 0.8, "flow": 0.7,
            }, self.settings)
            self.assertGreaterEqual(out.steering, -1.0)
            self.assertLessEqual(out.steering, 1.0)
            self.assertGreaterEqual(out.throttle, 0.0)
            self.assertLessEqual(out.throttle, 1.0)
            self.assertGreaterEqual(out.brake, 0.0)
            self.assertLessEqual(out.brake, 1.0)
            self.assertFalse(out.throttle > 0 and out.brake > 0)

    def test_straight_road_can_request_throttle(self):
        out = self.controller.update({
            "valid": True, "confidence": 0.95, "offset": 0.0,
            "curvature": 0.0, "flow": 0.0,
        }, self.settings)
        self.assertGreater(out.throttle, 0.0)
        self.assertEqual(out.brake, 0.0)


if __name__ == "__main__":
    unittest.main()
