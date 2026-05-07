"""Frame-based wrapper around FocusDataset.

:class:`FocusFrameDataset` wraps a :class:`~focus.data.base_dataset.FocusDataset`
and resolves the disk-resident frames for every sample's time segment.  Each
item exposes the :class:`~focus.data.data_models.Request` and
:class:`~focus.data.data_models.Reference` from the underlying dataset together
with the corresponding frame paths, the native video FPS, and the effective FPS
after stride subsampling.

The ``frames_folder`` in :class:`~focus.config.FocusConfig` determines which
set of pre-extracted frames is used (plain or overlayed).  To maintain both
side-by-side, run :class:`~focus.preprocessing.FrameExtractorPreprocessor`
into distinct folder names and pass the matching config here::

    from focus import FocusConfig, set_config
    from focus.data import FocusDataset, FocusFrameDataset
    from focus.enums import DatasetSplit, Track

    base = FocusDataset("heico", DatasetSplit.TEST, "original", Track.SEGMENT)

    overlay_ds = FocusFrameDataset(base, stride=25,
                                   config=FocusConfig(frames_folder="frames_overlay"))
    plain_ds   = FocusFrameDataset(base, stride=25,
                                   config=FocusConfig(frames_folder="frames"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import overload

from focus.config import DATASET_BASE_FPS, FocusConfig, get_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Reference, Request

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    """One sample from a :class:`FocusFrameDataset`.

    Attributes
    ----------
    request : Request
        The VQA input presented to the model.
    reference : Reference
        The ground-truth reference for evaluation.
    frame_paths : list[Path]
        Ordered list of frame file paths covering ``[start_time, end_time]``
        at the dataset stride.
    base_fps : float
        Native frame rate of the source video (frames per second).
    fps : float
        Effective frame rate after stride subsampling
        (``base_fps / stride``).
    """

    request: Request
    reference: Reference
    frame_paths: list[Path]
    base_fps: float
    fps: float


class FocusFrameDataset:
    """Frame-path wrapper around a :class:`FocusDataset`.

    For each sample in the wrapped dataset, resolves the paths of pre-extracted
    JPEG frames that fall within the sample's ``[start_time, end_time]`` window
    and returns them alongside the original request/reference pair.

    Parameters
    ----------
    dataset : FocusDataset
        The underlying dataset supplying requests and references.
    stride : int, optional
        Frame subsampling interval applied when computing frame paths.
        ``stride=1`` selects every pre-extracted frame; ``stride=25`` selects
        every 25th.  Default ``1``.
    config : FocusConfig or None, optional
        Configuration instance.  When ``None``, the global config is used.
        Pass a config with a custom ``frames_folder`` to select a specific
        pre-extracted frame set (e.g. plain vs. overlayed).

    Notes
    -----
    Frame paths are resolved from ``<dataset_dir>/<FRAMES_FOLDER>/<video_stem>/
    frame{index:07d}.jpg``, where the frame index advances by *stride* from
    ``round(start_time * base_fps)`` to ``round(end_time * base_fps)``
    (inclusive).  Frames must have been extracted beforehand by
    :class:`~focus.preprocessing.FrameExtractorPreprocessor`.

    ``base_fps`` is looked up from :data:`~focus.config.DATASET_BASE_FPS` — no
    video file needs to be open at dataset construction time.
    """

    def __init__(
        self,
        dataset: FocusDataset,
        stride: int = 25,
        config: FocusConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self.stride = max(stride, 1)
        self._cfg = config or get_config()

        root = self._cfg.get_dataset_dir(dataset=dataset.dataset)
        self._frames_root = root / self._cfg.FRAMES_FOLDER

        if dataset.dataset not in DATASET_BASE_FPS:
            raise ValueError(
                f"No base FPS defined for dataset {dataset.dataset!r}. "
                f"Add it to focus.config.DATASET_BASE_FPS."
            )
        self._base_fps = float(DATASET_BASE_FPS[dataset.dataset])
        self._fps = self._base_fps / self.stride

        logger.info(
            f"FocusFrameDataset: dataset={dataset.dataset!r}, "
            f"base_fps={self._base_fps}, stride={self.stride}, "
            f"effective_fps={self._fps:.4g}, frames_root={self._frames_root}."
        )

    # ── properties ────────────────────────────────────────────────────

    @property
    def base_fps(self) -> float:
        """Native frame rate of the source videos."""
        return self._base_fps

    @property
    def fps(self) -> float:
        """Effective frame rate after stride subsampling."""
        return self._fps

    # ── internals ─────────────────────────────────────────────────────

    def _get_single(self, idx: int) -> FrameSample:
        request, reference = self._dataset[idx]
        video_stem = Path(request.videoID).stem
        folder = self._frames_root / video_stem
        start_frame = round(request.start_time * self._base_fps)
        end_frame = round(request.end_time * self._base_fps)
        frame_paths = [
            folder / f"frame{i:07d}.jpg" for i in range(start_frame, end_frame + 1, self.stride)
        ]
        return FrameSample(
            request=request,
            reference=reference,
            frame_paths=frame_paths,
            base_fps=self._base_fps,
            fps=self._fps,
        )

    # ── sequence interface ────────────────────────────────────────────

    @overload
    def __getitem__(self, idx: int) -> FrameSample: ...

    @overload
    def __getitem__(self, idx: slice) -> list[FrameSample]: ...

    def __getitem__(self, idx: int | slice) -> FrameSample | list[FrameSample]:
        if isinstance(idx, slice):
            return [self._get_single(i) for i in range(*idx.indices(len(self)))]
        return self._get_single(idx)

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"FocusFrameDataset(dataset={self._dataset!r}, stride={self.stride})"
