"""
Diagnose why MediaPipeSource is getting zero valid frames.

Tests OpenCV camera capture and MediaPipe pose detection SEPARATELY,
so we can tell which one is actually failing instead of guessing.
Saves a captured frame to disk either way so you can look at it.

Usage:
    python3 diagnose_camera.py
    python3 diagnose_camera.py --camera 1        # try a different index
"""
import sys
import time

camera_idx = 0
if "--camera" in sys.argv:
    camera_idx = int(sys.argv[sys.argv.index("--camera") + 1])

print(f"=== Step 1: Can OpenCV open camera {camera_idx} at all? ===")
try:
    import cv2
except ImportError:
    print("opencv-python isn't installed. Run: pip install opencv-python")
    sys.exit(1)

print(f"OpenCV version: {cv2.__version__}")

cap = cv2.VideoCapture(camera_idx)
print(f"cap.isOpened(): {cap.isOpened()}")

if not cap.isOpened():
    print()
    print("Camera did not open at all. On Linux this is usually one of:")
    print("  - Wrong camera index — try: python3 diagnose_camera.py --camera 1")
    print("  - Permission issue — check you're in the 'video' group:")
    print("      groups $USER")
    print("      sudo usermod -aG video $USER   (then log out/in)")
    print("  - Another app is holding the camera (close Zoom/browser/etc.)")
    print("  - List available cameras: ls -la /dev/video*")
    sys.exit(1)

print()
print("=== Step 2: Can it actually READ frames? (trying 30 times) ===")
successes = 0
last_good_frame = None
for i in range(30):
    ret, frame = cap.read()
    if ret and frame is not None:
        successes += 1
        last_good_frame = frame
        mean_brightness = frame.mean()
        print(f"  read {i+1:2d}: OK  shape={frame.shape}  "
              f"mean_brightness={mean_brightness:.1f}"
              f"{'  <- looks BLACK/blank' if mean_brightness < 5 else ''}")
    else:
        print(f"  read {i+1:2d}: FAILED (ret={ret})")
    time.sleep(0.05)

cap.release()

print()
print(f"Result: {successes}/30 reads succeeded.")

if successes == 0:
    print()
    print("Camera opens but never delivers a frame. Try:")
    print("  - A different backend:")
    print("      cap = cv2.VideoCapture(camera_idx, cv2.CAP_V4L2)")
    print("  - Giving it a moment after opening before the first read")
    print("    (some webcams need ~0.5-1s warm-up)")
    print("  - A different USB port, if it's an external webcam")
    sys.exit(1)

if last_good_frame.mean() < 5:
    print()
    print("Frames are coming through but appear to be all-black.")
    print("Check the camera isn't physically covered/off, and that no")
    print("privacy shutter or OS-level camera toggle is active.")

import os
debug_path = os.path.join(os.getcwd(), "diagnose_camera_frame.jpg")
cv2.imwrite(debug_path, last_good_frame)
print(f"\nSaved a captured frame to: {debug_path}  <- open this and look at it")

print()
print("=== Step 3: Does MediaPipe detect a pose in that frame? ===")
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )
except ImportError as e:
    print(f"mediapipe import failed: {e}")
    sys.exit(1)

import urllib.request
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aura-m1")
os.makedirs(cache_dir, exist_ok=True)
model_path = os.path.join(cache_dir, "pose_landmarker_lite.task")
if not os.path.exists(model_path):
    print("Downloading pose model (first run only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        model_path,
    )

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE,   # single-image mode for this one-off test
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = PoseLandmarker.create_from_options(options)

img_rgb = cv2.cvtColor(last_good_frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
result = landmarker.detect(mp_image)
landmarker.close()

if result.pose_landmarks:
    print(f"SUCCESS: detected {len(result.pose_landmarks)} pose(s) with "
          f"{len(result.pose_landmarks[0])} landmarks.")
    print()
    print("Camera and MediaPipe both work — the issue in run_live.py is")
    print("likely just that no one was in frame yet, or lighting/distance")
    print("made detection harder in VIDEO mode's continuous tracking.")
    print("Stand fully in frame, well-lit, a few feet back, and try")
    print("run_live.py again.")
else:
    print("No pose detected in this frame.")
    print()
    print("Camera capture works fine — this is a DETECTION issue, not a")
    print(f"camera issue. Open {debug_path} and check:")
    print("  - Is a person actually visible and mostly in-frame?")
    print("  - Is it well lit? Pose detection struggles in low light.")
    print("  - Try standing further back — full torso should be visible.")
