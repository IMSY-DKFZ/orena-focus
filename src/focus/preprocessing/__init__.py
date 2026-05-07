"""Preprocessing utilities — frame extraction and timestamp overlays."""

from focus.preprocessing.frame_extraction import FrameExtractorPreprocessor
from focus.preprocessing.video_overlay import VideoTimestampOverlayPreprocessor


def extract_frames(
    dataset: str = "heico",
    stride: int = 1,
    max_workers: int = 2,
    use_overlay: bool = False,
    resolution: tuple[int, int] | None = None,
) -> None:
    """Extract JPEG frames from all videos in a FOCUS dataset.

    Parameters
    ----------
    dataset : str, optional
        FOCUS dataset identifier (default: ``"heico"``).
    stride : int, optional
        Frame skip interval — ``1`` extracts every frame, ``25`` extracts
        every 25th frame (default: ``1``).
    max_workers : int, optional
        Number of parallel worker processes (default: ``2``).
    use_overlay : bool, optional
        When ``True``, read from timestamp-overlaid videos where available
        (default: ``False``).
    resolution : tuple[int, int] or None, optional
        Optional ``(width, height)`` resize target (default: ``None``).
    """
    FrameExtractorPreprocessor(
        stride=stride,
        use_overlay=use_overlay,
        resolution=resolution,
    ).process(dataset=dataset, max_workers=max_workers)


def overlay_timestamps(
    dataset: str = "heico",
    text_scale: float = 1.5,
    max_workers: int = 2,
) -> None:
    """Burn ``hh:mm:ss`` timestamps into all videos in a FOCUS dataset.

    Parameters
    ----------
    dataset : str, optional
        FOCUS dataset identifier (default: ``"heico"``).
    text_scale : float, optional
        Font scale factor for the rendered timestamp (default: ``1.5``).
    max_workers : int, optional
        Number of parallel worker processes (default: ``2``).
    """
    VideoTimestampOverlayPreprocessor(text_scale=text_scale).process(
        dataset=dataset, max_workers=max_workers
    )


__all__ = [
    "extract_frames",
    "overlay_timestamps",
    "FrameExtractorPreprocessor",
    "VideoTimestampOverlayPreprocessor",
]
