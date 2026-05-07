"""Burn visible timestamps into FOCUS surgical videos.

Each frame receives a ``hh:mm:ss`` text overlay derived from its position in
the video.  The overlaid copies are written to the dataset's ``overlayed/``
folder as MP4 files and are used by :class:`~focus.preprocessing.FrameExtractor`
when ``use_overlay=True``.

Motivation
----------
The PROCEDURE track of the ORena SAVE FOCUS challenge asks VLMs questions that
require long-horizon temporal reasoning over full laparoscopic procedures (e.g.
"when was the sponge last retrieved?").  Because many VLMs cannot infer elapsed
time from frame content alone, burning an explicit clock into every frame gives
the model a reliable temporal anchor without modifying the challenge evaluation
protocol.
"""

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from progiter import ProgIter

from focus.config import VIDEO_SUFFIXES, get_config
from focus.data.download import get_unpreprocessed_dataset

logger = logging.getLogger(__name__)


class VideoTimestampOverlayPreprocessor:
    """Burn a ``hh:mm:ss`` timestamp into every frame of a surgical video.

    The rendered text appearance is configurable at construction time and is
    shared across all videos processed by a single instance.

    Parameters
    ----------
    text_org : tuple[int, int], optional
        ``(x, y)`` pixel coordinate of the bottom-left corner of the timestamp
        text, in OpenCV convention (origin at top-left). Default ``(20, 50)``.
    text_font_face : int, optional
        OpenCV font identifier (``cv2.FONT_*``). Default
        ``cv2.FONT_HERSHEY_SIMPLEX``.
    text_color : tuple[int, int, int], optional
        BGR colour of the timestamp text. Default ``(255, 255, 255)`` (white).
    text_line_type : int, optional
        OpenCV line type used when drawing text. Default ``cv2.LINE_AA``
        (anti-aliased).
    text_scale : float, optional
        Font scale factor passed to ``cv2.putText``. Stroke thickness is
        derived as ``max(1, int(text_scale * 2))``. Default ``1.5``.

    Notes
    -----
    :meth:`process` resolves the output directory through the global
    :class:`~focus.config.FocusConfig` (via :func:`~focus.config.get_config`).
    To produce and keep multiple overlay variants side-by-side (e.g. different
    text scales or positions), set a distinct ``overlay_folder`` name on the
    config before each run — no re-download required::

        from focus import FocusConfig, set_config
        from focus.preprocessing import VideoTimestampOverlayPreprocessor

        set_config(FocusConfig(overlay_folder="overlayed_large_text"))
        VideoTimestampOverlayPreprocessor(text_scale=2.5).process()

        set_config(FocusConfig(overlay_folder="overlayed_small_text"))
        VideoTimestampOverlayPreprocessor(text_scale=1.0).process()

    Both variants end up as named sub-folders inside the same dataset directory,
    alongside the original ``videos/`` folder.
    """

    def __init__(
        self,
        text_org: tuple[int, int] = (20, 50),
        text_font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        text_color: tuple[int, int, int] = (255, 255, 255),
        text_line_type: int = cv2.LINE_AA,
        text_scale: float = 1.5,
    ):
        self.text_org = text_org
        self.text_font_face = text_font_face
        self.text_color = text_color
        self.text_line_type = text_line_type
        self.text_scale = text_scale

    def _process_video(self, args: tuple[Path, Path]) -> tuple[bool, str]:
        """Overlay timestamps on a single video and write the result to disk.

        Reads ``video_path`` frame-by-frame, computes the elapsed time from the
        frame index and the video's FPS, draws the ``hh:mm:ss`` string into the
        frame, and writes the annotated frame to ``target_path`` as an MP4.
        The source file is not modified.

        Parameters
        ----------
        args : tuple[Path, Path]
            ``(video_path, target_path)`` where *video_path* is the source video
            and *target_path* is the destination MP4 file.

        Returns
        -------
        tuple[bool, str]
            ``(True, success_message)`` on success, ``(False, error_message)``
            if any exception is raised.
        """
        video_path, target_path = args
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 25.0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(str(target_path), fourcc, fps, (frame_width, frame_height))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                total_secs = int(frame_idx / fps)
                h, m, s = (
                    total_secs // 3600,
                    (total_secs % 3600) // 60,
                    total_secs % 60,
                )
                timestamp = f"{h:02d}:{m:02d}:{s:02d}"

                cv2.putText(
                    img=frame,
                    text=timestamp,
                    org=self.text_org,
                    fontFace=self.text_font_face,
                    fontScale=self.text_scale,
                    color=self.text_color,
                    thickness=max(1, int(self.text_scale * 2)),
                    lineType=self.text_line_type,
                )

                out.write(frame)
                frame_idx += 1

            cap.release()
            out.release()
            return True, f"Success: {video_path.name} ({frame_count} frames)"

        except Exception as e:
            return False, f"Error: {video_path.name} - {str(e)}"

    def process(
        self,
        dataset: str = "heico",
        max_workers: int = 2,
        videos_list: list[str] | None = None,
    ) -> None:
        """Overlay timestamps on all videos in a FOCUS dataset.

        Loads the dataset metadata, resolves which video files are both present
        on disk and referenced by the metadata, and dispatches per-video
        processing to a ``ProcessPoolExecutor``.  Results are written to the
        ``overlayed/`` sub-folder of the dataset directory as
        ``<stem>_overlay.mp4`` files.

        Parameters
        ----------
        dataset : str, optional
            FOCUS dataset identifier (e.g. ``"heico"``). Must be a key in
            ``focus.config.FOCUS_DATASETS``. Default ``"heico"``.
        max_workers : int, optional
            Maximum number of parallel worker processes. Increase for faster
            throughput on machines with many CPU cores, but note that each
            worker holds one open ``VideoCapture`` and one ``VideoWriter``.
            Default ``2``.
        videos_list : list[str] or None, optional
            If provided, only videos whose file names (e.g.
            ``"procedure_001.avi"``) appear in *videos_list* are processed.
            When ``None`` (default), all videos referenced in the dataset
            metadata are processed.

        Raises
        ------
        RuntimeError
            If the dataset has not been downloaded yet (propagated from
            :func:`~focus.data.download.get_unpreprocessed_dataset`).

        Warnings
        --------
        UserWarning
            If the overlay output folder already contains files, which will be
            overwritten by this run.
        """
        ds = get_unpreprocessed_dataset(dataset=dataset, split="all")
        cfg = get_config()
        local_dir = cfg.get_dataset_dir(dataset=dataset)
        videos_folder = local_dir / cfg.VIDEO_FOLDER
        overlay_folder = local_dir / cfg.OVERLAY_FOLDER

        if overlay_folder.exists() and any(overlay_folder.iterdir()):
            warnings.warn(
                f"Overlay folder {overlay_folder} already exists and will be overwritten."
            )
        overlay_folder.mkdir(exist_ok=True)

        videos_present = {
            p for p in videos_folder.iterdir() if p.is_file() and p.suffix in VIDEO_SUFFIXES
        }
        videos_requested = set()

        for sample in ProgIter(ds, desc="Reading in videos"):
            videos_requested.add(videos_folder / dict(sample)["video"])

        logger.info(
            f"Dataset {dataset!r} has {len(videos_present)} videos present and "
            f"{len(videos_requested)} mentioned in metadata."
        )

        pp_videos = videos_present.intersection(videos_requested)

        if videos_list:
            videos_list_names = set(videos_list)
            pp_videos = {vp for vp in pp_videos if vp.name in videos_list_names}

        if not pp_videos:
            logger.warning(
                "No videos to process after intersection. Check that all videos are downloaded and "
                "videos_list paths are correct."
            )
            return

        logger.info(f"Processing {len(pp_videos)} videos...")

        tasks = [(vp, overlay_folder / (vp.stem + "_overlay.mp4")) for vp in pp_videos]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                ProgIter(
                    executor.map(self._process_video, tasks),
                    total=len(tasks),
                )
            )

        successes = [mes for s, mes in results if s]
        errors = [mes for s, mes in results if not s]

        logger.info(f"Finished: {len(successes)} successful, {len(errors)} errors.")
        for mes in successes:
            logger.debug(mes)
        for mes in errors:
            logger.error(mes)
