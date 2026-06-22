"""Environment-based configuration for data paths and settings.

Configure paths via environment variables:

    export FOCUS_ROOT_DIR=.../data/focus/

Or in code:

    from focus import FocusConfig, set_config
    cfg = FocusConfig(root_dir=".../data/focus")
    set_config(cfg)
"""

from __future__ import annotations

import os
from pathlib import Path

from focus.enums import Track

FOCUS_DATASETS = {
    "heico": "orena-dkfz/heico-focus-vqa",
    "lapchole": "orena-dkfz/lapchole-focus-vqa"
}

DATASET_BASE_FPS = {
    "heico": 25,
    "lapchole": 30,
}

# Maximum allowed response latency (seconds) per challenge track.  Responses
# slower than this limit are treated as incorrect by the Evaluator.  Resolved
# via the ``track`` argument of :meth:`focus.evaluation.Evaluator.run`.
TRACK_MAX_LATENCY: dict[Track, float] = {
    Track.FRAME: 5.0,
    Track.SEGMENT: 15.0,
    Track.PROCEDURE: 30.0,
}

_ENV_ROOT_DIR = "FOCUS_ROOT_DIR"

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


class FocusConfig:
    """Central configuration for all FOCUS data paths.

    Reads from environment variables by default, but accepts explicit overrides.

    Parameters
    ----------
    root_dir : Path or str or None, optional
        Root directory that contains one sub-folder per dataset (e.g.
        ``root_dir/heico/``).  When ``None``, the value of the
        ``FOCUS_ROOT_DIR`` environment variable is used.
    overlay_folder : str or None, optional
        Name of the sub-folder inside each dataset directory where timestamp-
        overlaid videos are stored.  Defaults to ``"overlayed"``.  Set this to
        a custom name to keep multiple overlay variants side-by-side without
        re-downloading the dataset::

            set_config(FocusConfig(overlay_folder="overlayed_large_text"))

    frames_folder : str or None, optional
        Name of the sub-folder inside each dataset directory where extracted
        frames are stored.  Defaults to ``"frames"``.  Set this to a custom
        name to maintain multiple frame extraction configurations in parallel::

            set_config(FocusConfig(frames_folder="frames_1fps"))
    """

    OVERLAY_FOLDER = "overlayed"
    FRAMES_FOLDER = "frames"
    VIDEO_FOLDER = "videos"

    def __init__(
        self,
        root_dir: Path | str | None = None,
        overlay_folder: str | None = None,
        frames_folder: str | None = None,
    ):
        if root_dir is None:
            root_dir = os.environ.get(_ENV_ROOT_DIR)
            if not root_dir:
                raise ValueError(
                    f"Either provide root_dir or set the {_ENV_ROOT_DIR} environment variable."
                )
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise ValueError(f"Root dir {self.root_dir.name} does not exist.")
        if overlay_folder is not None:
            self.OVERLAY_FOLDER = overlay_folder
        if frames_folder is not None:
            self.FRAMES_FOLDER = frames_folder

    def get_dataset_dir(self, dataset: str = "heico") -> Path:
        if dataset not in FOCUS_DATASETS:
            raise ValueError(f"{dataset} is not a FOCUS dataset.")
        p = self.root_dir / dataset
        p.mkdir(exist_ok=True)
        return p


# Module-level singleton, lazily initialised from environment variables.
_global_config: FocusConfig | None = None


def get_config() -> FocusConfig:
    """Return the global FocusConfig (created once from env vars)."""
    global _global_config
    if _global_config is None:
        _global_config = FocusConfig()
    return _global_config


def set_config(config: FocusConfig) -> None:
    """Override the global FocusConfig."""
    global _global_config
    _global_config = config
