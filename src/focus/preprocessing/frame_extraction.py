"""Extract JPEG frames from FOCUS surgical videos.

Frames are written to per-video sub-folders inside the dataset's ``frames/``
directory and consumed by :class:`~focus.dataset.FocusDataset` in
``mode="preextracted"``.  When ``use_overlay=True`` the preprocessor reads
from the timestamp-overlaid copies produced by
:class:`~focus.preprocessing.VideoTimestampOverlayPreprocessor` instead of the
raw source videos.

Notes
-----
To maintain multiple frame extraction configurations side-by-side (e.g.
different strides or resolutions), set a distinct ``frames_folder`` name on
the config before each run — no re-download required::

    from focus import FocusConfig, set_config
    from focus.preprocessing import FrameExtractorPreprocessor

    set_config(FocusConfig(frames_folder="frames_1fps"))
    FrameExtractorPreprocessor(stride=25).process()

    set_config(FocusConfig(frames_folder="frames_5fps"))
    FrameExtractorPreprocessor(stride=5).process()

Both sets of frames end up as named sub-folders inside the same dataset
directory, alongside the original ``videos/`` folder.
"""

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import decord
from progiter import ProgIter

from focus.config import VIDEO_SUFFIXES, get_config
from focus.data.download import get_unpreprocessed_dataset

logger = logging.getLogger(__name__)


class FrameExtractorPreprocessor:
    """Extract JPEG frames from FOCUS dataset videos at a fixed stride.

    Parameters
    ----------
    stride : int, optional
        Frame skip interval.  ``stride=1`` extracts every frame;
        ``stride=25`` extracts every 25th frame (1 fps for a 25 fps source).
        Default ``1``.
    use_overlay : bool, optional
        When ``True``, frames are extracted exclusively from timestamp-overlaid
        copies (see
        :class:`~focus.preprocessing.VideoTimestampOverlayPreprocessor`).
        Videos for which no overlay exists are skipped with a ``UserWarning``
        and their names are logged at ``DEBUG`` level.  Default ``False``.
    resolution : tuple[int, int] or None, optional
        Target ``(width, height)`` in pixels.  When provided, every extracted
        frame is resized with ``cv2.INTER_AREA`` before being saved.  When
        ``None`` (default), frames are saved at their native resolution.

    Notes
    -----
    :meth:`process` resolves the output directory through the global
    :class:`~focus.config.FocusConfig` (via :func:`~focus.config.get_config`).
    To maintain multiple frame extraction configurations side-by-side, set a
    distinct ``frames_folder`` on the config before each run — see the module-
    level docstring for an example.
    """

    def __init__(
        self,
        stride: int = 1,
        use_overlay: bool = False,
        resolution: tuple[int, int] | None = None,
    ):
        self.stride = stride
        self.use_overlay = use_overlay
        self.resolution = resolution

    def _extract_video(self, args: tuple[Path, Path]) -> tuple[bool, str]:
        """Extract frames from a single video and write them to disk.

        Reads every :attr:`stride`-th frame from *video_path* using
        ``decord``, optionally resizes each frame to :attr:`resolution`, and
        saves them as JPEG files named ``frame{index:07d}.jpg`` inside
        *output_dir*.

        Parameters
        ----------
        args : tuple[Path, Path]
            ``(video_path, output_dir)`` where *video_path* is the source
            video and *output_dir* is the per-video output folder (created if
            absent).

        Returns
        -------
        tuple[bool, str]
            ``(True, success_message)`` on success, ``(False, error_message)``
            if any exception is raised.
        """
        video_path, output_dir = args
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            stride = max(self.stride, 1)
            indices = list(range(0, len(vr), stride))
            batch_size = 32

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                frames = vr.get_batch(batch_indices)
                if hasattr(frames, "asnumpy"):
                    frames = frames.asnumpy()

                for j, frame in enumerate(frames):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if self.resolution:
                        frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
                    filepath = output_dir / f"frame{i + j:07d}.jpg"
                    cv2.imwrite(str(filepath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            return True, f"Success: {video_path.name} ({len(indices)} frames extracted)"

        except Exception as e:
            return False, f"Error: {video_path.name} - {str(e)}"

    def process(
        self,
        dataset: str = "heico",
        max_workers: int = 2,
        videos_list: list[str] | None = None,
    ) -> None:
        """Extract frames from all videos in a FOCUS dataset.

        Loads the dataset metadata, resolves which video files are both present
        on disk and referenced by the metadata, and dispatches per-video frame
        extraction to a ``ProcessPoolExecutor``.  Frames are written to
        per-video sub-folders inside the dataset's frames directory, named
        ``<video_stem>/frame{index:07d}.jpg``.  The frame skip interval and
        output resolution are taken from :attr:`stride` and :attr:`resolution`.

        Parameters
        ----------
        dataset : str, optional
            FOCUS dataset identifier (e.g. ``"heico"``).  Must be a key in
            ``focus.config.FOCUS_DATASETS``.  Default ``"heico"``.
        max_workers : int, optional
            Maximum number of parallel worker processes.  Each worker holds
            one open ``decord.VideoReader``.  Default ``2``.
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
            If the frames output folder already contains files, which will be
            overwritten by this run.
        """
        decord.bridge.set_bridge("native")

        ds = get_unpreprocessed_dataset(dataset=dataset, split="all")
        cfg = get_config()
        local_dir = cfg.get_dataset_dir(dataset=dataset)
        videos_folder = local_dir / cfg.VIDEO_FOLDER
        overlay_folder = local_dir / cfg.OVERLAY_FOLDER
        frames_folder = local_dir / cfg.FRAMES_FOLDER

        if frames_folder.exists() and any(frames_folder.iterdir()):
            warnings.warn(f"Frames folder {frames_folder} already exists and will be overwritten.")
        frames_folder.mkdir(parents=True, exist_ok=True)

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
                "No videos to process after intersection. "
                "Check that all videos are downloaded and videos_list paths are correct."
            )
            return

        tasks = []
        if self.use_overlay:
            missing_overlay = []
            for vp in pp_videos:
                overlay_candidate = overlay_folder / (vp.stem + "_overlay.mp4")
                if overlay_candidate.exists():
                    tasks.append((overlay_candidate, frames_folder / vp.stem))
                else:
                    missing_overlay.append(vp.name)

            if missing_overlay:
                warnings.warn(
                    f"{len(missing_overlay)} video(s) have no overlay and will be skipped. "
                    f"Run VideoTimestampOverlayPreprocessor first, or set use_overlay=False."
                )
                for name in missing_overlay:
                    logger.debug(f"No overlay found, skipping: {name}")
        else:
            tasks = [(vp, frames_folder / vp.stem) for vp in pp_videos]

        if not tasks:
            logger.warning("No videos to process. Aborting.")
            return

        logger.info(f"Processing {len(tasks)}/{len(pp_videos)} videos (stride={self.stride})...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                ProgIter(
                    executor.map(self._extract_video, tasks),
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
