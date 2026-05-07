"""Download FOCUS video data from HuggingFace Hub.

QA annotations (parquet files) are fetched automatically by
``datasets.load_dataset`` when :class:`~focus.data.base_dataset.FocusDataset`
is constructed.  This module handles the download of the accompanying video
files, which are too large for the HuggingFace datasets cache.
"""

import logging

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from focus.config import FOCUS_DATASETS, get_config

logger = logging.getLogger(__name__)


def download(dataset: str = "heico", revision: str | None = None) -> None:
    """Download a FOCUS dataset's video files from HuggingFace Hub.

    Video files are stored in ``<root_dir>/<dataset>/videos/``.  QA annotations
    (parquet) are fetched on demand by :class:`~focus.data.base_dataset.FocusDataset`
    and do not need to be downloaded separately.

    Repeated calls are safe — HuggingFace caches downloaded files and skips
    files that are already present locally.

    Parameters
    ----------
    dataset : str, optional
        Dataset identifier (default: ``"heico"``).
    revision : str or None, optional
        HuggingFace repository revision — a git tag (e.g. ``"v1.0"``), branch,
        or commit hash.  Should match the ``revision`` passed to
        :class:`~focus.data.base_dataset.FocusDataset` so that videos and
        annotations come from the same release.  ``None`` uses the default branch.

    Raises
    ------
    ValueError
        If no HuggingFace repository is configured for the given dataset.

    Examples
    --------
    >>> download("heico")  # doctest: +SKIP
    """
    cfg = get_config()
    local_dir = cfg.get_dataset_dir(dataset=dataset)
    repo_id = FOCUS_DATASETS.get(dataset)
    if not repo_id:
        raise ValueError(
            f"No HuggingFace repository configured for dataset {dataset!r}. "
            f"Available datasets: {', '.join(FOCUS_DATASETS.keys())}"
        )
    logger.info(f"Downloading {dataset!r} videos from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        revision=revision,
        ignore_patterns=["data/*.parquet"],
        allow_patterns=["videos/*"],
    )
    logger.info("Download complete.")


def get_unpreprocessed_dataset(dataset: str, split: str = "all") -> Dataset:
    """Load all QA annotations for a FOCUS dataset from HuggingFace Hub.

    Used internally by preprocessing steps (frame extraction, timestamp overlay)
    to determine which videos are referenced in the dataset.

    Parameters
    ----------
    dataset : str
        Dataset identifier (e.g., ``"heico"``).
    split : {"train", "test", "all"}, optional
        Which annotations to load.  ``"all"`` loads both train and test rows
        (default: ``"all"``).

    Returns
    -------
    Dataset
        HuggingFace Dataset object containing the loaded samples.

    Raises
    ------
    ValueError
        If no HuggingFace repository is configured for the given dataset.
    """
    repo_id = FOCUS_DATASETS.get(dataset)
    if not repo_id:
        raise ValueError(
            f"No HuggingFace repository configured for dataset {dataset!r}. "
            f"Available datasets: {', '.join(FOCUS_DATASETS.keys())}"
        )
    hf_split = "train+test" if split == "all" else split
    return load_dataset(repo_id, "all_tracks", split=hf_split)
