"""Video-clip wrapper around FocusDataset.

:class:`FocusVideoDataset` wraps a :class:`~focus.data.base_dataset.FocusDataset`
and produces a temporary MP4 clip for each sample's ``[start_time, end_time]``
window on demand.  The clip is written to ``/tmp`` via
:func:`tempfile.NamedTemporaryFile` and the caller is responsible for deleting
it afterwards::

    from focus.data import FocusDataset, FocusVideoDataset
    from focus.enums import DatasetSplit, Track

    base = FocusDataset("heico", DatasetSplit.TEST, "original", Track.SEGMENT)
    vds  = FocusVideoDataset(base, stride=5, use_overlay=True)

    sample = vds[0]
    try:
        feed_to_model(sample.video_path)
    finally:
        sample.video_path.unlink(missing_ok=True)

Clips are re-encoded with the ``mp4v`` codec at ``base_fps / stride`` frames
per second, making the file compact while preserving wall-clock duration.
Optionally resize each frame to ``resolution`` before writing.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import cv2
import decord

from focus.config import FocusConfig, get_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import Reference, Request

logger = logging.getLogger(__name__)


@dataclass
class VideoSample:
    """One sample from a :class:`FocusVideoDataset`.

    Attributes
    ----------
    request : Request
        The VQA input presented to the model.
    reference : Reference
        The ground-truth reference for evaluation.
    video_path : Path
        Path to the temporary MP4 clip covering ``[start_time, end_time]``.
        The caller must delete this file when it is no longer needed.
    base_fps : float
        Native frame rate of the source video (frames per second).
    fps : float
        Effective frame rate of the written clip
        (``base_fps / stride``).
    """

    request: Request
    reference: Reference
    video_path: Path
    base_fps: float
    fps: float


class FocusVideoDataset:
    """Temporary-clip wrapper around a :class:`FocusDataset`.

    For each sample, extracts the relevant video segment from the source
    video on disk, subsamples it by *stride*, and writes a compact temporary
    MP4 to ``/tmp``.

    Parameters
    ----------
    dataset : FocusDataset
        The underlying dataset supplying requests and references.
    stride : int, optional
        Frame subsampling interval.  ``stride=1`` keeps every frame;
        ``stride=5`` keeps every 5th frame and reduces the clip length and
        file size proportionally.  Default ``1``.
    use_overlay : bool, optional
        When ``True``, clips are sourced from the timestamp-overlaid videos in
        ``cfg.OVERLAY_FOLDER`` (produced by
        :class:`~focus.preprocessing.VideoTimestampOverlayPreprocessor`).
        When ``False`` (default), the original videos in ``cfg.VIDEO_FOLDER``
        are used.
    resolution : tuple[int, int] or None, optional
        Target ``(width, height)`` in pixels.  When provided, every frame is
        resized with ``cv2.INTER_AREA`` before being written.  When ``None``
        (default), frames are written at their native resolution.
    config : FocusConfig or None, optional
        Configuration instance.  When ``None``, the global config is used.

    Notes
    -----
    Each :meth:`__getitem__` call opens the source video with ``decord``,
    batch-reads the segment frames, and writes a new MP4 via
    ``cv2.VideoWriter``.  The resulting file is kept in ``/tmp``; callers
    must call ``sample.video_path.unlink(missing_ok=True)`` when done.

    ``base_fps`` is read from the source video via ``decord.VideoReader``
    and cached per video to avoid repeated file opens across samples that
    share the same procedure.
    """

    def __init__(
        self,
        dataset: FocusDataset,
        stride: int = 25,
        use_overlay: bool = False,
        resolution: tuple[int, int] | None = None,
        config: FocusConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self.stride = max(stride, 1)
        self.use_overlay = use_overlay
        self.resolution = resolution
        self._cfg = config or get_config()

        root = self._cfg.get_dataset_dir(dataset=dataset.dataset)
        video_folder = self._cfg.OVERLAY_FOLDER if use_overlay else self._cfg.VIDEO_FOLDER
        self._video_root = root / video_folder
        self._fps_cache: dict[str, float] = {}

        logger.info(
            f"FocusVideoDataset: dataset={dataset.dataset!r}, "
            f"stride={self.stride}, use_overlay={use_overlay}, "
            f"video_root={self._video_root}."
        )

    # ── internals ─────────────────────────────────────────────────────

    def _video_path(self, video_id: str) -> Path:
        if self.use_overlay:
            return self._video_root / f"{Path(video_id).stem}_overlay.mp4"
        return self._video_root / video_id

    def _make_clip(self, request: Request) -> tuple[Path, float, float]:
        """Extract the sample segment into a temporary MP4.

        Returns
        -------
        tuple[Path, float, float]
            ``(clip_path, base_fps, fps)`` where *clip_path* is the temporary
            file, *base_fps* is the native video frame rate, and *fps* is the
            effective rate of the written clip.
        """
        video_path = self._video_path(request.videoID)
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
        base_fps = float(vr.get_avg_fps())
        self._fps_cache[request.videoID] = base_fps  # cache under videoID, not overlay name
        fps = base_fps / self.stride

        start_frame = round(request.start_time * base_fps)
        end_frame = min(round(request.end_time * base_fps), len(vr) - 1)
        indices = list(range(start_frame, end_frame + 1, self.stride))
        if not indices:
            indices = [start_frame]

        frames = vr.get_batch(indices).asnumpy()
        del vr

        orig_h, orig_w = frames.shape[1], frames.shape[2]
        out_size = self.resolution if self.resolution else (orig_w, orig_h)

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp")
        tmp_path = Path(tmp.name)
        tmp.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, out_size)
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.resolution:
                frame_bgr = cv2.resize(frame_bgr, out_size, interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)
        writer.release()

        return tmp_path, base_fps, fps

    def _get_single(self, idx: int) -> VideoSample:
        request, reference = self._dataset[idx]
        video_path, base_fps, fps = self._make_clip(request)
        return VideoSample(
            request=request,
            reference=reference,
            video_path=video_path,
            base_fps=base_fps,
            fps=fps,
        )

    # ── sequence interface ────────────────────────────────────────────

    @overload
    def __getitem__(self, idx: int) -> VideoSample: ...

    @overload
    def __getitem__(self, idx: slice) -> list[VideoSample]: ...

    def __getitem__(self, idx: int | slice) -> VideoSample | list[VideoSample]:
        if isinstance(idx, slice):
            return [self._get_single(i) for i in range(*idx.indices(len(self)))]
        return self._get_single(idx)

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"FocusVideoDataset("
            f"dataset={self._dataset!r}, "
            f"stride={self.stride}, "
            f"use_overlay={self.use_overlay})"
        )
