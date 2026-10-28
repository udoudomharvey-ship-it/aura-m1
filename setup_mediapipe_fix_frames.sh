#!/usr/bin/env bash
# Run from the ROOT of your aura-m1 project.
#   bash setup_mediapipe_fix_frames.sh
#
# Fixes a real bug found in your last run: "Total frames: 0" happened
# because the very first camera read or pose detection came back
# empty, and the base SkeletonSource.frames() treats any None as
# "end of stream" — correct for FileSource/SyntheticSource (finite
# recordings), wrong for a live camera, where a momentary missed
# read or "no person detected" frame is completely normal.
#
# This overrides frames() in MediaPipeSource specifically: it now
# skips transient misses (up to 60 in a row, ~2s) instead of ending
# the session, and only stops for real once the camera is genuinely
# gone. Everything else in the file is unchanged from the previous fix.

set -euo pipefail

echo "Writing aura_sdk/sensors/sources.py ..."
cat > aura_sdk/sensors/sources.py << 'AURA_EOF_SOURCES'
"""
aura_sdk/sensors/sources.py
─────────────────────────────
Language: Python
Why: Sensor drivers need OS-level camera access, file I/O,
     and the MediaPipe / OpenCV ecosystem — all Python-native.
     The joint positions produced here are passed to Rust for
     entropy computation. Python handles acquisition; Rust handles math.

Supported sources
──────────────────
  MediaPipeSource  — Standard webcam via Google MediaPipe pose estimation
  FileSource       — Replay pre-recorded skeleton sequences from CSV/JSON
  SyntheticSource  — Deterministic synthetic skeleton (testing / demo)
  KinectV2Source   — Azure Kinect via pyk4a (when hardware available)

All sources implement the same SkeletonSource interface so the
pipeline never needs to know which hardware is connected.
"""

from __future__ import annotations
import csv
import json
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


# ── Joint type ─────────────────────────────────────────────────────────────

@dataclass
class SkeletonFrame:
    """
    One frame of skeleton data.
    joints: list of (x, y, z) tuples in metres, spine-base normalised.
    timestamp_ms: wall-clock time when frame was captured.
    """
    joints:       list[tuple[float, float, float]]
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    frame_idx:    int   = 0
    source:       str   = "unknown"

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    def normalised(self, spine_idx: int = 0) -> "SkeletonFrame":
        """Return a new frame with joints normalised to spine-base origin."""
        if not self.joints:
            return self
        sx, sy, sz = self.joints[spine_idx]
        norm_joints = [(x - sx, y - sy, z - sz) for x, y, z in self.joints]
        return SkeletonFrame(
            joints=norm_joints,
            timestamp_ms=self.timestamp_ms,
            frame_idx=self.frame_idx,
            source=self.source,
        )


# ── Abstract base ──────────────────────────────────────────────────────────

class SkeletonSource(ABC):
    """
    Abstract skeleton data source.
    All concrete sources implement this interface.
    """

    @abstractmethod
    def open(self) -> None:
        """Initialise the source (open camera, load file, etc.)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @abstractmethod
    def read_frame(self) -> Optional[SkeletonFrame]:
        """
        Read one skeleton frame.
        Returns None if no frame is available (end of file, camera error).
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Returns True if the source is ready to produce frames."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    def frames(self) -> Iterator[SkeletonFrame]:
        """Iterator over all available frames."""
        while self.is_open():
            frame = self.read_frame()
            if frame is None:
                break
            yield frame


# ── Synthetic source ───────────────────────────────────────────────────────

class SyntheticSource(SkeletonSource):
    """
    Deterministic synthetic skeleton source for testing and demos.
    Generates 120 frames matching HCI-Lab-30 workload trajectory.

    No hardware required. No dependencies.
    Produces the same sequence every run (deterministic seed).
    """

    N_JOINTS = 25

    def __init__(self, n_frames: int = 120, seed: int = 42, fps: float = 30.0):
        self.n_frames  = n_frames
        self.fps       = fps
        self._seed     = seed
        self._frame    = 0
        self._q        = 0.68
        self._open     = False
        self._rng_state: int = seed

    def _lcg(self) -> float:
        """Deterministic LCG — no random module needed."""
        self._rng_state = (self._rng_state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        return self._rng_state / 0xFFFFFFFF

    def _next_frame_joints(self) -> list[tuple[float, float, float]]:
        task   = "T1" if self._frame < 40 else ("T2" if self._frame < 80 else "T3")
        target = {"T1": 0.72, "T2": 0.50, "T3": 0.28}[task]
        noise  = self._lcg() * 0.05 - 0.025
        self._q = max(0.05, min(0.95,
            self._q + (target - self._q) * 0.10 + noise
        ))
        spread = (1.0 - self._q) * 0.3 + 0.05
        joints = []
        for j in range(self.N_JOINTS):
            x = (j * 0.1 + self._lcg() * spread) * spread
            y = (j * 0.05 + self._lcg() * spread) * spread
            z = (j * 0.08 + self._lcg() * spread) * spread
            joints.append((x, y, z))
        return joints

    def open(self)  -> None: self._open = True
    def close(self) -> None: self._open = False
    def is_open(self) -> bool:
        return self._open and self._frame < self.n_frames

    def read_frame(self) -> Optional[SkeletonFrame]:
        if not self.is_open():
            return None
        joints = self._next_frame_joints()
        frame  = SkeletonFrame(
            joints=joints,
            frame_idx=self._frame,
            source="synthetic",
        )
        self._frame += 1
        return frame.normalised()

    def __repr__(self):
        return (f"SyntheticSource(n_frames={self.n_frames}, "
                f"fps={self.fps}, seed={self._seed})")


# ── File source — CSV or JSON replay ──────────────────────────────────────

class FileSource(SkeletonSource):
    """
    Replay pre-recorded skeleton sequences from CSV or JSON.

    CSV format (one row per joint per frame):
        frame_idx, joint_idx, x, y, z

    JSON format:
        {"frames": [{"joints": [[x,y,z], ...]}, ...]}

    Used for:
      - Offline validation against HCI-Lab-30 recordings
      - Regression testing after SDK updates
      - Demos without a camera
    """

    def __init__(self, path: str, normalise: bool = True):
        self.path       = path
        self.normalise  = normalise
        self._frames:   list[SkeletonFrame] = []
        self._idx:      int  = 0
        self._open_flag: bool = False

    def open(self) -> None:
        path = self.path.lower()
        if path.endswith(".json"):
            self._load_json()
        elif path.endswith(".csv"):
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file format: {self.path}")
        self._open_flag = True

    def _load_json(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        raw_frames = data.get("frames", [])
        for i, fr in enumerate(raw_frames):
            joints = [tuple(j) for j in fr["joints"]]
            frame  = SkeletonFrame(joints=joints, frame_idx=i, source=self.path)
            self._frames.append(frame.normalised() if self.normalise else frame)

    def _load_csv(self) -> None:
        raw: dict[int, list] = {}
        with open(self.path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fi = int(row["frame_idx"])
                ji = int(row["joint_idx"])
                if fi not in raw:
                    raw[fi] = []
                # Pad with None placeholders if joints arrive out of order
                while len(raw[fi]) <= ji:
                    raw[fi].append(None)
                raw[fi][ji] = (float(row["x"]), float(row["y"]), float(row["z"]))
        for fi in sorted(raw.keys()):
            joints = [j for j in raw[fi] if j is not None]
            frame  = SkeletonFrame(joints=joints, frame_idx=fi, source=self.path)
            self._frames.append(frame.normalised() if self.normalise else frame)

    def close(self) -> None:
        self._open_flag = False

    def is_open(self) -> bool:
        return self._open_flag and self._idx < len(self._frames)

    def read_frame(self) -> Optional[SkeletonFrame]:
        if not self.is_open():
            return None
        frame = self._frames[self._idx]
        self._idx += 1
        return frame

    def __len__(self) -> int:
        return len(self._frames)

    def __repr__(self):
        return f"FileSource(path='{self.path}', frames={len(self._frames)})"


# ── MediaPipe source — live webcam ─────────────────────────────────────────

class MediaPipeSource(SkeletonSource):
    """
    Live skeleton tracking via Google MediaPipe's Pose Landmarker task.
    Uses a standard webcam — no specialist hardware required.

    Requires: pip install mediapipe opencv-python

    IMPORTANT: this uses the MediaPipe TASKS API
    (mediapipe.tasks.python.vision.PoseLandmarker), not the old
    `mediapipe.solutions.pose` API. Google ended support for the
    legacy `solutions` API in March 2023, and — as of the mediapipe
    versions available for Python 3.13 (0.10.30+) — importing it
    raises `AttributeError: module 'mediapipe' has no attribute
    'solutions'` outright; there is no working pinned version for
    current Python releases. The Tasks API is the only supported,
    actively maintained path, and produces the same 33-point BlazePose
    landmark topology, so the joint mapping below (established against
    the old API) is unchanged.

    On first use, this downloads a small (.task) pose model bundle to
    a local cache directory if one isn't already there or explicitly
    provided via `model_path` — that step needs internet access once;
    after that it's fully offline.

    Joint mapping (MediaPipe → Aura-M1 indices):
        0: nose            → head
        1–4: eye landmarks → face region (mapped to index 1-4)
        11–16: shoulders, elbows, wrists (bilateral)
        23–30: hips, knees, ankles, heels (bilateral)
        Spine-base: midpoint of hips (computed)
    """

    # MediaPipe landmark indices mapped to 25-joint Aura-M1 ordering
    _LANDMARK_MAP = [
        0,   # 0 head (nose)
        11,  # 1 left shoulder
        12,  # 2 right shoulder
        13,  # 3 left elbow
        14,  # 4 right elbow
        15,  # 5 left wrist
        16,  # 6 right wrist
        17,  # 7 left pinky
        18,  # 8 right pinky
        19,  # 9 left index
        20,  # 10 right index
        23,  # 11 left hip
        24,  # 12 right hip
        25,  # 13 left knee
        26,  # 14 right knee
        27,  # 15 left ankle
        28,  # 16 right ankle
        29,  # 17 left heel
        30,  # 18 right heel
        31,  # 19 left foot index
        32,  # 20 right foot index
        7,   # 21 left ear
        8,   # 22 right ear
        9,   # 23 left eye
        10,  # 24 right eye
    ]

    # Google's official model bundle storage — "lite" is the fastest/
    # smallest variant, matching this project's low-power design ethos.
    # "full" and "heavy" trade speed for accuracy; pass model_complexity
    # to use one of those instead.
    _MODEL_URLS = {
        "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }

    def __init__(self, camera_idx: int = 0, width: int = 640,
                 height: int = 480, min_confidence: float = 0.5,
                 model_complexity: str = "lite", model_path: Optional[str] = None):
        self.camera_idx        = camera_idx
        self.width              = width
        self.height             = height
        self.min_confidence     = min_confidence
        self.model_complexity   = model_complexity
        self.model_path         = model_path
        self._cap                = None
        self._landmarker         = None
        self._frame_idx          = 0
        self._last_timestamp_ms  = -1

    def _resolve_model_path(self) -> str:
        """Returns a local path to the .task model bundle, downloading
        it to a cache directory on first use if `model_path` wasn't
        given explicitly."""
        if self.model_path is not None:
            return self.model_path

        import os
        import urllib.request

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "aura-m1")
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"pose_landmarker_{self.model_complexity}.task"
        cache_path = os.path.join(cache_dir, filename)

        if not os.path.exists(cache_path):
            url = self._MODEL_URLS.get(self.model_complexity)
            if url is None:
                raise ValueError(
                    f"Unknown model_complexity '{self.model_complexity}' — "
                    f"expected one of {list(self._MODEL_URLS)}."
                )
            print(f"[MediaPipeSource] Downloading pose model to {cache_path} ...")
            urllib.request.urlretrieve(url, cache_path)

        return cache_path

    def open(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode,
            )
        except ImportError:
            raise ImportError(
                "MediaPipe source requires:\n"
                "  pip install mediapipe opencv-python\n"
                "Or use SyntheticSource for testing without a camera."
            )

        model_path = self._resolve_model_path()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

        self._cap = cv2.VideoCapture(self.camera_idx)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open camera {self.camera_idx}")

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Optional[SkeletonFrame]:
        if not self.is_open():
            return None
        import time
        import cv2
        import mediapipe as mp

        ret, img = self._cap.read()
        if not ret:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # detect_for_video requires strictly increasing timestamps —
        # wall-clock milliseconds satisfy that in normal use, but the
        # explicit max() guards against two frames landing on the same
        # millisecond at high frame rates.
        timestamp_ms = max(int(time.time() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]  # first detected person

        # Extract 25 joints using landmark map
        joints = []
        for mp_idx in self._LANDMARK_MAP:
            if mp_idx < len(lm):
                pt = lm[mp_idx]
                # MediaPipe coords are normalised [0,1] — convert to metres
                # Approximate: 1 unit ≈ 1 metre for typical standing distance
                joints.append((pt.x - 0.5, pt.y - 0.5, pt.z))
            else:
                joints.append((0.0, 0.0, 0.0))

        # Normalise to the spine-base origin as the docstring above
        # promises ("Spine-base: midpoint of hips (computed)"). The
        # generic SkeletonFrame.normalised() defaults to spine_idx=0,
        # but index 0 in _LANDMARK_MAP is the head (nose), not the
        # spine base — calling normalised() with no argument here
        # silently re-centred every joint on the head instead of the
        # hips, which is inconsistent with HEMM's H_sp derivation
        # (Sec 3.3.1) and with every other SkeletonSource. Indices 11
        # and 12 are left/right hip (see _LANDMARK_MAP), so the true
        # spine-base is their midpoint; that midpoint is subtracted
        # from every joint directly instead.
        hx = (joints[11][0] + joints[12][0]) / 2.0
        hy = (joints[11][1] + joints[12][1]) / 2.0
        hz = (joints[11][2] + joints[12][2]) / 2.0
        joints = [(x - hx, y - hy, z - hz) for x, y, z in joints]

        frame = SkeletonFrame(
            joints=joints,
            frame_idx=self._frame_idx,
            source="mediapipe",
        )
        self._frame_idx += 1
        return frame

    def __repr__(self):
        return (f"MediaPipeSource(camera={self.camera_idx}, "
                f"resolution={self.width}×{self.height}, "
                f"model={self.model_complexity})")

    # How many consecutive missed reads (failed camera grab, or no
    # person detected in frame) to tolerate before giving up — ~2s at
    # 30fps. Momentary misses are completely normal for a live camera
    # (someone steps out of frame briefly, a frame drops) and must not
    # end the session; a camera that's truly gone dark still needs to
    # stop eventually rather than spin forever.
    _MAX_CONSECUTIVE_MISSES = 60

    def frames(self) -> Iterator[SkeletonFrame]:
        """
        Overrides SkeletonSource.frames(). The base implementation
        treats a single None from read_frame() as "end of stream" —
        correct for FileSource/SyntheticSource, where None means the
        recording is over. For a LIVE camera, None more often means
        "this particular frame had a failed grab or no detected pose,"
        which is routine and should be skipped, not treated as the
        stream ending. Only genuinely losing the camera (is_open()
        going False, or too many misses in a row) ends it here.
        """
        misses = 0
        while self.is_open():
            frame = self.read_frame()
            if frame is None:
                misses += 1
                if misses >= self._MAX_CONSECUTIVE_MISSES:
                    print(f"[MediaPipeSource] No valid frame for "
                          f"{misses} consecutive reads — stopping.")
                    break
                continue
            misses = 0
            yield frame
AURA_EOF_SOURCES

echo ""
echo "Done. Running the full test suite now..."
python3 -m unittest discover -s aura_sdk_tests -v
