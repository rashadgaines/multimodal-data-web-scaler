"""Core pipeline: ingest -> filter -> extract -> store."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open_clip
import ray
import torch
import webdataset as wds
import yt_dlp

from cosmos_curator.config import Config
from cosmos_curator.extractors import (
    classify,
    embed,
    extract_audio,
    extract_frames,
    transcribe,
)
from cosmos_curator.filters import filter_duplicate, filter_relevance, filter_technical

logger = logging.getLogger(__name__)


@dataclass
class Video:
    """Single data structure that flows through the pipeline."""

    url: str
    local_path: Path | None = None
    metadata: dict = field(default_factory=dict)
    frames: list[np.ndarray] = field(default_factory=list)
    audio: np.ndarray | None = None
    transcript: str = ""
    embeddings: np.ndarray | None = None
    labels: dict = field(default_factory=dict)


def download(url: str, temp_dir: Path) -> Video:
    """Download video using yt-dlp."""
    video = Video(url=url)

    temp_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(temp_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height>=720]+bestaudio/best[height>=720]/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "ignoreerrors": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                video.metadata["title"] = info.get("title", "")
                video.metadata["uploader"] = info.get("uploader", "")
                video.metadata["upload_date"] = info.get("upload_date", "")
                video.metadata["view_count"] = info.get("view_count", 0)
                video.metadata["video_id"] = info.get("id", "")

                ext = info.get("ext", "mp4")
                video_id = info.get("id", "video")
                local_path = temp_dir / f"{video_id}.{ext}"
                if local_path.exists():
                    video.local_path = local_path
                else:
                    for p in temp_dir.glob(f"{video_id}.*"):
                        if p.suffix in (".mp4", ".webm", ".mkv"):
                            video.local_path = p
                            break
    except Exception as e:
        video.metadata["download_error"] = str(e)
        logger.warning(f"Download failed for {url}: {e}")

    return video


class PipelineProcessor:
    """Stateful processor for Ray distributed execution."""

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seen_hashes: set[str] = set()

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            config.clip_model, pretrained=config.clip_pretrained
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer(config.clip_model)

        logger.info(f"Initialized pipeline processor on {self.device}")

    def process(self, url: str) -> Video | None:
        """Process a single video through the full pipeline."""
        logger.debug(f"Processing: {url}")

        video = download(url, self.config.temp_dir)
        if video.local_path is None:
            logger.debug(f"Skipped (download failed): {url}")
            return None

        video = filter_technical(
            video,
            min_resolution=self.config.min_resolution,
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
        )
        if video is None:
            logger.debug(f"Skipped (technical filter): {url}")
            return None

        video = extract_frames(video, fps=self.config.frame_fps)
        if not video.frames:
            logger.debug(f"Skipped (no frames): {url}")
            return None

        video = filter_relevance(
            video,
            clip_model=self.clip_model,
            clip_tokenizer=self.clip_tokenizer,
            clip_preprocess=self.clip_preprocess,
            threshold=self.config.clip_threshold,
            device=self.device,
        )
        if video is None:
            logger.debug(f"Skipped (relevance filter): {url}")
            return None

        video = filter_duplicate(video, self.seen_hashes)
        if video is None:
            logger.debug(f"Skipped (duplicate): {url}")
            return None

        video = extract_audio(video)
        video = transcribe(video, model_size=self.config.whisper_model)
        video = embed(video, self.clip_model, self.clip_preprocess, self.device)
        video = classify(video)

        logger.info(f"Processed: {video.metadata.get('title', url)}")
        return video


def write_video_to_shard(video: Video, shard_writer: wds.ShardWriter, index: int) -> None:
    """Write a single video to a WebDataset shard."""
    key = f"{index:06d}"

    sample = {
        "__key__": key,
        "json": json.dumps({
            "url": video.url,
            "metadata": video.metadata,
            "labels": video.labels,
        }).encode(),
        "txt": video.transcript.encode(),
    }

    if video.local_path and video.local_path.exists():
        sample["mp4"] = video.local_path.read_bytes()

    if video.embeddings is not None:
        sample["npy"] = video.embeddings.tobytes()

    shard_writer.write(sample)


def process_dataset(config: Config) -> int:
    """Process the full dataset using Ray Data.

    Returns:
        Number of videos successfully processed.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    urls = config.get_urls()
    logger.info(f"Processing {len(urls)} URLs")

    if config.ray_address:
        ray.init(address=config.ray_address)
    else:
        ray.init(ignore_reinit_error=True)

    ds = ray.data.from_items(urls)
    processor = PipelineProcessor(config)

    def process_batch(batch: dict) -> dict:
        results = []
        for url in batch["item"]:
            video = processor.process(url)
            results.append(video)
        return {"video": results}

    processed = ds.map_batches(
        process_batch,
        batch_size=1,
        num_cpus=1,
        num_gpus=0.25 if torch.cuda.is_available() else 0,
    )

    shard_pattern = str(config.output_dir / "shard-%06d.tar")
    idx = 0

    with wds.ShardWriter(shard_pattern, maxsize=config.shard_size) as sink:
        for batch in processed.iter_batches(batch_size=1):
            for video in batch["video"]:
                if video is not None:
                    write_video_to_shard(video, sink, idx)
                    idx += 1
                    if video.local_path and video.local_path.exists():
                        video.local_path.unlink()

    logger.info(f"Processed {idx} videos to {config.output_dir}")

    shutil.rmtree(config.temp_dir, ignore_errors=True)
    ray.shutdown()

    return idx
