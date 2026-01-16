"""Cosmos Curator - Cosmology video curation pipeline."""

__version__ = "0.1.0"

from cosmos_curator.config import Config
from cosmos_curator.pipeline import Video, process_dataset

__all__ = ["Config", "Video", "process_dataset", "__version__"]
