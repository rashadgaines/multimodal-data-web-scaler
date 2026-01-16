#!/usr/bin/env python3
"""CLI entry point for the cosmology video curation pipeline."""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from cosmos_curator.config import Config
from cosmos_curator.pipeline import process_dataset

app = typer.Typer(
    name="cosmos-curator",
    help="Cosmology Video Curation Pipeline - curate astronomy videos at scale",
    add_completion=False,
)


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@app.command()
def main(
    urls: Optional[Path] = typer.Option(
        None, "--urls", "-u",
        help="Path to file containing video URLs (one per line)",
    ),
    output: Path = typer.Option(
        Path("./output"), "--output", "-o",
        help="Output directory for WebDataset shards",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to YAML configuration file",
    ),
    min_resolution: int = typer.Option(
        720, "--min-res",
        help="Minimum video resolution (height in pixels)",
    ),
    min_duration: int = typer.Option(
        10, "--min-dur",
        help="Minimum video duration in seconds",
    ),
    max_duration: int = typer.Option(
        1800, "--max-dur",
        help="Maximum video duration in seconds",
    ),
    clip_threshold: float = typer.Option(
        0.25, "--clip-thresh",
        help="CLIP similarity threshold for relevance filtering",
    ),
    ray_address: Optional[str] = typer.Option(
        None, "--ray",
        help="Ray cluster address (e.g., 'auto' or 'ray://host:port')",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose logging",
    ),
) -> None:
    """Process videos through the cosmology curation pipeline.

    Downloads videos, filters by quality and relevance, extracts frames/audio,
    transcribes speech, generates embeddings, and outputs WebDataset shards.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    if config:
        logger.info(f"Loading config from {config}")
        cfg = Config.from_yaml(config)
    elif urls:
        cfg = Config(
            input_urls=urls,
            output_dir=output,
            min_resolution=min_resolution,
            min_duration=min_duration,
            max_duration=max_duration,
            clip_threshold=clip_threshold,
            ray_address=ray_address,
        )
    else:
        typer.echo("Error: Either --urls or --config must be provided", err=True)
        raise typer.Exit(1)

    logger.info(f"Input: {cfg.input_urls}")
    logger.info(f"Output: {cfg.output_dir}")

    try:
        count = process_dataset(cfg)
        logger.info(f"Pipeline complete! Processed {count} videos.")
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
