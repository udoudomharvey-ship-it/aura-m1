"""
aura_sdk.sensors
─────────────────
Language: Python
Skeleton data sources for the Aura-M1 pipeline.

    from aura_sdk.sensors import (
        SyntheticSource, FileSource, MediaPipeSource,
        SkeletonFrame, SkeletonSource,
    )
"""
from .sources import (
    SkeletonSource,
    SkeletonFrame,
    SyntheticSource,
    FileSource,
    MediaPipeSource,
)

__all__ = [
    "SkeletonSource", "SkeletonFrame",
    "SyntheticSource", "FileSource", "MediaPipeSource",
]
