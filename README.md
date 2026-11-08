# Baku F1 Autopilot

Windows screen-based driving assistant for Roblox Formula Apex Racing.

## Baseline
- Car: 2006 Rouiller
- Brake bias: 55 (leave the in-game setting unchanged)
- Controls: W/A/S/D
- No DRS or ERS
- Intended track: Baku

## Current controller
- Captures the monitor live; no screenshot upload is required.
- Uses multi-slice road-corridor vision with an adaptive, bounded PID-style steering controller.
- Adds smoothed steering, bend feed-forward, relative optical-flow estimation, conservative corner braking, and line-recovery braking.
- Enforces mutually exclusive throttle/brake requests and releases controls when road confidence is too low.
- Adds automated controller safety tests to the Windows build pipeline.
- Reduces analysis resolution to limit CPU use.
- Sends keyboard input only while a Roblox window is the foreground window.
- F8 is the emergency stop. F6 starts/records lap times manually. F7 records a user-reported incident.
- Saves lap times, best time, events, and bounded controller tuning values in `%APPDATA%/BakuAutopilot/learning.json`.

## Important limitations
This is still a visual heuristic, not a trained racing agent or game integration. Road segmentation can mistake shadows, walls, kerbs, or scenery for asphalt. The controller does not read actual speed, gear, steering angle, damage, track position, or the game's lap counter. Braking is estimated from visible road curvature, lateral offset, and relative optical flow; optical flow is not a real speed reading and the braking thresholds are not calibrated to the Rouiller's physics. The controller has not yet been validated on live Baku footage. Lap timing is manual with F6. Visual lap detection is experimental and can miss the line or trigger on a similar-looking scene; verify every recorded time. The app does not automatically identify crashes or read actual speed/gear telemetry, and saved incident reports are user-entered. Test in a private session, observe first, and keep F8 available. No fully autonomous, fast, or crash-free lap is guaranteed. Test in observe-only mode first and use a private server; this controller is experimental and may misread scenery as track.

## Run
1. Install Python 3.11 on Windows.
2. Run `pip install -r requirements.txt`.
3. Run `python main.py`.
4. Open Roblox and put the car on track.
5. Start observing and verify the preview.
6. Focus the Roblox window, then enable autopilot.
7. Press F8 to stop. At the start/finish line, press F6 once to arm the visual checkpoint; the app will try to record each return automatically; F7 to save an incident.

## Build
GitHub Actions builds a standalone Windows executable and a portable folder. Open **Actions → Build Windows app → latest successful run → Artifacts**.
