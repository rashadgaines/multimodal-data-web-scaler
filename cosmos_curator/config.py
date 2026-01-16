"""Configuration for the cosmology video curation pipeline."""

import tempfile
from pathlib import Path

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Pipeline configuration."""

    input_urls: Path | list[str]
    output_dir: Path
    min_resolution: int = 720
    min_duration: int = 10
    max_duration: int = 1800
    clip_threshold: float = 0.25
    frame_fps: float = 1.0
    shard_size: int = 1_000_000_000  # 1GB
    ray_address: str | None = None
    temp_dir: Path = Path(tempfile.gettempdir()) / "cosmos_curator"
    num_workers: int = 4
    whisper_model: str = "base"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    @field_validator("input_urls", mode="before")
    @classmethod
    def parse_input_urls(cls, v: str | Path | list[str]) -> Path | list[str]:
        if isinstance(v, list):
            return v
        return Path(v)

    @field_validator("output_dir", "temp_dir", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        return Path(v)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_urls(self) -> list[str]:
        """Get list of URLs from input_urls."""
        if isinstance(self.input_urls, list):
            return self.input_urls
        with open(self.input_urls) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
