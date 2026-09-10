"""Application-owned deterministic data pipeline."""

from pathlib import Path
from typing import Any, Dict


PIPELINE_READY = False


def build(
    contract: Dict[str, Any],
    staging_directory: Path,
    published_directory: Path,
) -> Dict[str, Any]:
    """Build frozen artifacts and return a dataset-manifest.v1 object."""
    raise RuntimeError(
        "Compile the application workflows, then implement data/pipeline.py and set "
        "PIPELINE_READY = True."
    )
