"""
aura_sdk_tools/skeleton_visualizer.py
────────────────────────────────────────
A standalone PyVista desktop viewer for Aura-M1: renders the 25-joint
skeleton stream with bones, colors the whole figure by the live BNN
state (1=worst workload -> 5=best), and overlays Q_HCI / entropy
readouts as on-screen text.

Requirements
─────────────
    pip install pyvista

NOT part of the on-device pipeline -- this is a desktop debug/demo tool
only. PyVista needs a real GPU/OpenGL context and has nothing to do
with Aura-M1's ultra-low-power embedded target.

⚠ Joint ordering assumption
────────────────────────────
HEMMExtractor's docstring says joints must be "25 (Kinect v2) or 20
(Kinect v1)" but nowhere in the codebase have I seen confirmation of
the actual joint INDEX order. BONE_CONNECTIONS below uses the
standard Microsoft Kinect v2 Body joint order:

    0 SpineBase     5 ElbowLeft     10 WristRight    15 FootLeft
    1 SpineMid      6 WristLeft     11 HandRight      16 HipRight
    2 Neck          7 HandLeft      12 HipLeft        17 KneeRight
    3 Head          8 ShoulderRight 13 KneeLeft        18 AnkleRight
    4 ShoulderLeft  9 ElbowRight    14 AnkleLeft        19 FootRight
    20 SpineShoulder  21 HandTipLeft  22 ThumbLeft
    23 HandTipRight   24 ThumbRight

If your real joint source (MediaPipeSource / KinectV2Source / FileSource)
uses a different order, edit BONE_CONNECTIONS to match -- the skeleton
will still render (right point count) but bones will connect the wrong
joints until this is fixed.

On synthetic sources (SyntheticSource / ScenarioSyntheticSource): these
generate mathematically-placed points, not real body proportions, so
you'll see a moving point cloud with correct color/state behavior but
it won't look like a person. Use FileSource or a live camera source
for a visually convincing demo.

Usage
─────
    python3 skeleton_visualizer.py                     # live camera, if available
    python3 skeleton_visualizer.py --synthetic          # SyntheticSource demo
    python3 skeleton_visualizer.py --scenario           # ScenarioSyntheticSource, all 3 reachable classes
    python3 skeleton_visualizer.py --file path/to.csv   # FileSource replay
"""
from __future__ import annotations
import argparse
import sys
import time

import numpy as np
import pyvista as pv

from aura_sdk.hemm.extractor import HEMMExtractor
from aura_sdk.inference.bnn import AuraBNN


# Standard Kinect v2 Body joint bone connectivity (25 joints).
# See module docstring re: verifying this matches your actual joint order.
BONE_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),                     # spine -> neck -> head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (6, 22),    # left arm + hand
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (10, 24),  # right arm + hand
    (0, 12), (12, 13), (13, 14), (14, 15),                # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),                # right leg
]

# State 1 (worst workload) -> 5 (best): red -> yellow -> green gradient.
STATE_COLORS = {
    1: (0.85, 0.15, 0.15),
    2: (0.90, 0.45, 0.15),
    3: (0.90, 0.80, 0.15),
    4: (0.55, 0.80, 0.25),
    5: (0.20, 0.75, 0.35),
}


def make_source(args):
    """Builds whichever SkeletonSource the CLI args asked for."""
    if args.scenario:
        from aura_sdk_tests.scenario_synthetic_source import ScenarioSyntheticSource
        from aura_sdk.recommend import ProblemClass
        return ScenarioSyntheticSource(
            phases=[
                (ProblemClass.SPATIAL_RESTLESSNESS, 60),
                (ProblemClass.COGNITIVE_OVERLOAD, 60),
                (ProblemClass.SUSTAINED_LOW_QUALITY, 60),
            ],
            seed=args.seed,
        )
    if args.synthetic:
        from aura_sdk.sensors.sources import SyntheticSource
        return SyntheticSource(n_frames=args.n_frames, seed=args.seed)
    if args.file:
        from aura_sdk.sensors.sources import FileSource
        return FileSource(args.file)
    from aura_sdk.sensors.sources import MediaPipeSource
    return MediaPipeSource()


class SkeletonVisualizer:
    def __init__(self, source, fps: float = 30.0):
        self.source = source
        self.extractor = HEMMExtractor()
        self.bnn = AuraBNN()
        self.frame_interval = 1.0 / fps

        self.plotter = pv.Plotter()
        self.plotter.set_background("black")
        self.plotter.add_axes()

        self._joint_actor = None
        self._bone_actor = None
        self._text_actor = None
        self._prev_frame = None
        self._camera_framed = False

    def _bones_polydata(self, points: np.ndarray) -> pv.PolyData:
        lines = []
        for a, b in BONE_CONNECTIONS:
            if a < len(points) and b < len(points):
                lines.append([2, a, b])
        mesh = pv.PolyData(points)
        mesh.lines = np.hstack(lines) if lines else np.array([], dtype=int)
        return mesh

    def _render_frame(self, points: np.ndarray, color: tuple, hud_text: str):
        if self._joint_actor is not None:
            self.plotter.remove_actor(self._joint_actor)
        if self._bone_actor is not None:
            self.plotter.remove_actor(self._bone_actor)
        if self._text_actor is not None:
            self.plotter.remove_actor(self._text_actor)

        cloud = pv.PolyData(points)
        self._joint_actor = self.plotter.add_mesh(
            cloud, color=color, point_size=14, render_points_as_spheres=True,
        )
        bones = self._bones_polydata(points)
        self._bone_actor = self.plotter.add_mesh(bones, color=color, line_width=4)
        self._text_actor = self.plotter.add_text(
            hud_text, position="upper_left", font_size=12, color="white",
        )

    def run(self):
        self.source.open()
        self.plotter.show(title="Aura-M1 Skeleton Visualizer", auto_close=False, interactive_update=True)

        try:
            for frame in self.source.frames():
                points = np.array(frame.joints, dtype=float)

                if self._prev_frame is not None:
                    feats = self.extractor.compute(self._prev_frame.joints, frame.joints)
                    result = self.bnn.infer(feats.h_sp, feats.h_te, feats.h_in)
                    color = STATE_COLORS.get(result.state, (0.7, 0.7, 0.7))
                    hud = (
                        f"state: {result.state}   Q_HCI: {result.q_hci:.3f}\n"
                        f"H_sp: {feats.h_sp:.3f}   H_te: {feats.h_te:.3f}   H_in: {feats.h_in:.3f}\n"
                        f"frame: {frame.frame_idx}"
                    )
                else:
                    color = (0.7, 0.7, 0.7)
                    hud = f"frame: {frame.frame_idx}  (warming up...)"

                self._render_frame(points, color, hud)
                self.plotter.reset_camera()
                self.plotter.update()
                time.sleep(self.frame_interval)

                self._prev_frame = frame
        except KeyboardInterrupt:
            pass
        finally:
            self.source.close()
            self.plotter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--synthetic", action="store_true", help="Use SyntheticSource")
    parser.add_argument("--scenario", action="store_true", help="Use ScenarioSyntheticSource (3-class demo)")
    parser.add_argument("--file", type=str, default=None, help="Path to a FileSource CSV/JSON")
    parser.add_argument("--n-frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    source = make_source(args)
    viz = SkeletonVisualizer(source, fps=args.fps)
    viz.run()


if __name__ == "__main__":
    sys.exit(main())
