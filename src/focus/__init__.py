"""focus-challenge — Utilities for the ORena SAVE FOCUS challenge.

Foreign Object Contextual Understanding for Safe Surgical AI.

All core types are available at the top level::

    from focus import (
        # Taxonomy
        Capability,
        # Foreign object types
        FOType, FO_DEFINITION, SPONGE, CLIP, SPECIMEN_BAG, SILICONE_LOOP,
        EXTERNAL_DRAIN, NEEDLE, GALLSTONE, SPECIMEN, MESH,
        # Answer formats
        Binary, Number, FOClass, OpenEnded, Matching, MultipleChoice, Time, Percentage,
        get_format_class,
        # Data classes
        Request, Reference, Response,
        # Serialisation helpers
        save_items, load_requests, load_references, load_responses,
        # Dataset
        FocusDataset, download,
        # Enumerations
        DatasetSplit, Track,
        # Configuration
        FocusConfig, get_config, set_config,
        # Evaluation
        Evaluator,
    )
"""

from focus.config import FocusConfig, get_config, set_config
from focus.data.base_dataset import FocusDataset
from focus.data.data_models import (
    Reference,
    Request,
    Response,
    load_references,
    load_requests,
    load_responses,
    save_items,
)
from focus.data.download import download
from focus.data.formats import (
    Binary,
    FOClass,
    Matching,
    MultipleChoice,
    Number,
    OpenEnded,
    Percentage,
    Time,
    get_format_class,
)
from focus.enums import DatasetSplit, Track
from focus.evaluation import Evaluator
from focus.foreign_objects import (
    CLIP,
    EXTERNAL_DRAIN,
    FO_DEFINITION,
    FO_DEFINITIONS_FILE,
    GALLSTONE,
    MESH,
    NEEDLE,
    SILICONE_LOOP,
    SPECIMEN,
    SPECIMEN_BAG,
    SPONGE,
    FOType,
)
from focus.taxonomy import Capability

__version__ = "0.2.0"

import logging as _logging

_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
del _logging

__all__ = [
    # Taxonomy
    "Capability",
    # Foreign object types
    "FOType",
    "FO_DEFINITION",
    "FO_DEFINITIONS_FILE",
    "SPONGE",
    "CLIP",
    "SPECIMEN_BAG",
    "SILICONE_LOOP",
    "EXTERNAL_DRAIN",
    "NEEDLE",
    "GALLSTONE",
    "SPECIMEN",
    "MESH",
    # Answer formats
    "Binary",
    "Number",
    "Percentage",
    "FOClass",
    "OpenEnded",
    "Matching",
    "MultipleChoice",
    "Time",
    "get_format_class",
    # Data classes
    "Request",
    "Reference",
    "Response",
    # Serialisation
    "save_items",
    "load_requests",
    "load_references",
    "load_responses",
    # Dataset
    "FocusDataset",
    "download",
    # Enumerations
    "DatasetSplit",
    "Track",
    # Configuration
    "FocusConfig",
    "get_config",
    "set_config",
    # Evaluation
    "Evaluator",
]
