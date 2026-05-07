"""Data preparation example for the ORena SAVE FOCUS challenge.

This script walks through the full preparation pipeline:

  1. Configure the library (root directory)
  2. Download the dataset videos from HuggingFace Hub
  3. Burn timestamps into the source videos (optional)
  4. Extract JPEG frames from the videos

QA annotations (parquet files) are fetched automatically when you
construct a :class:`~focus.FocusDataset` — no separate step needed.

Run each step independently — every step is idempotent and safe to re-run.

Prerequisites
-------------
    pip install orena-focus

Set ``FOCUS_ROOT_DIR`` to an existing directory with enough disk space, or
pass it explicitly to ``FocusConfig`` as shown below.
"""

# ── 1. Configure ─────────────────────────────────────────────────────

from focus import FocusConfig, download, set_config
from focus.preprocessing import FrameExtractorPreprocessor, VideoTimestampOverlayPreprocessor

# Point the library at a local root directory.  Every dataset is stored as a
# sub-folder inside this root (e.g. <root>/heico/).
# Alternatively, just set the FOCUS_ROOT_DIR environment variable.

set_config(FocusConfig(root_dir="/data/focus"))

DATASET = "heico"

# ── 2. Download ───────────────────────────────────────────────────────

# Downloads videos from HuggingFace Hub into <root>/heico/videos/.
# QA annotation parquet files are streamed on demand — no manual step needed.
# Safe to call repeatedly — skips the download if already complete.
download(DATASET)

# ── 3. Timestamp overlay (optional) ──────────────────────────────────

# Burns a visible hh:mm:ss counter into each video and writes the result to
# <root>/heico/overlayed/.  Skip this step if you do not need overlayed videos.
VideoTimestampOverlayPreprocessor().process(dataset=DATASET, max_workers=4)

# ── 4. Frame extraction ───────────────────────────────────────────────

# Extract every frame (stride=1) from the original videos into
# <root>/heico/frames/<video_stem>/frame{index:07d}.jpg.
FrameExtractorPreprocessor(stride=1).process(dataset=DATASET, max_workers=4)

# To extract from the overlayed videos instead, use a separate frames folder
# so both variants live side-by-side:
set_config(FocusConfig(root_dir="/data/focus", frames_folder="frames_overlay"))
FrameExtractorPreprocessor(stride=1, use_overlay=True).process(dataset=DATASET, max_workers=4)
set_config(FocusConfig(root_dir="/data/focus"))  # restore default

# ── Done ──────────────────────────────────────────────────────────────
# The dataset is now ready.  Load it with FocusDataset:
#
#   from focus import FocusDataset, DatasetSplit, Track
#
#   ds = FocusDataset("heico", DatasetSplit.TEST, Track.SEGMENT)
#   request, reference = ds[0]
