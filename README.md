# Baku F1 Autopilot

Windows desktop prototype for hands-free keyboard driving in Roblox Formula Apex Racing.

## Baseline
- Car: 2006 Rouiller
- Brake bias: 55 (leave the in-game setting unchanged)
- Controls: W/A/S/D
- DRS / ERS: not used
- Target: Baku, repeated laps, improve consistency and lap time

## Important limitations
This is a screen-based controller, not a Roblox game integration. It observes the visible game window and sends ordinary keyboard input. It cannot know the game's physics, track coordinates, speed, or lap state unless those are visibly displayed. The initial vision controller is deliberately conservative and needs testing in the actual game. It does not promise a fast or crash-free lap out of the box.

Use only in a private session or where automation is permitted by the game's rules. Keep the game window visible. The Stop button and F8 emergency stop release all keys immediately.

## Run from source
1. Install Python 3.11 on Windows.
2. Run `pip install -r requirements.txt`.
3. Run `python main.py`.
4. Focus Roblox, line up on track, then press Start in the app.
5. Press F8 at any time to stop.

The app starts in **observe-only mode**. Enable keyboard control in the UI after checking the preview and confirming the correct Roblox window is selected.

## Learning
Lap observations and manual notes are saved in `%APPDATA%/BakuAutopilot/learning.json`. The starter learner records outcomes and tunes conservative steering/throttle limits; genuine track-specific optimization requires repeated real-game testing and visible lap-time/sector feedback.

## Build
GitHub Actions builds a Windows one-folder executable and uploads it as a workflow artifact. Open **Actions → Build Windows app → latest successful run → Artifacts**.
