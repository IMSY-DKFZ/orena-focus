"""Shared enumerations for the FOCUS challenge library."""

from enum import Enum


class DatasetSplit(str, Enum):
    """Dataset partitions used across all FOCUS datasets.

    Inherits from ``str`` so values can be passed directly to
    HuggingFace ``load_dataset`` split arguments.
    """

    TRAIN = "train"
    TEST = "test"
    ALL = "all"


class Track(str, Enum):
    """The three evaluation tracks of the ORena SAVE FOCUS challenge.

    Tracks increase in temporal and contextual complexity:

    * **FRAME** — single-image understanding: foreign object detection,
      identification, attribute recognition, and spatial localisation.
    * **SEGMENT** — short video segments (up to 5 min): motion-aware
      perception, short-term tracking, and brief action sequence understanding.
    * **PROCEDURE** — full laparoscopic procedures: long-horizon memory,
      persistent object tracking, aggregation over time, and global reasoning.
    """

    FRAME = "frame"
    SEGMENT = "segment"
    PROCEDURE = "procedure"
