"""Quality and relevance filters for video curation (pure functions)."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import imagehash
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from cosmos_curator.pipeline import Video

COSMOS_PROMPTS = [
    "a photograph of a galaxy",
    "a nebula in space",
    "astronomical observation",
    "cosmological simulation",
    "telescope image of stars",
    "visualization of the universe",
    "black hole rendering",
    "cosmic microwave background",
]


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def filter_technical(
    video: Video,
    min_resolution: int = 720,
    min_duration: int = 10,
    max_duration: int = 1800,
) -> Video | None:
    """Filter based on technical quality: resolution and duration."""
    if video.local_path is None:
        return None

    info = get_video_info(str(video.local_path))
    if not info:
        return None

    # Get duration
    duration = float(info.get("format", {}).get("duration", 0))
    if duration < min_duration or duration > max_duration:
        return None

    # Get resolution from video stream
    video_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        return None

    height = video_streams[0].get("height", 0)
    if height < min_resolution:
        return None

    # Store metadata
    video.metadata["duration"] = duration
    video.metadata["height"] = height
    video.metadata["width"] = video_streams[0].get("width", 0)

    return video


def filter_relevance(
    video: Video,
    clip_model: torch.nn.Module,
    clip_tokenizer,
    clip_preprocess,
    threshold: float = 0.25,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Video | None:
    """Filter based on CLIP similarity to cosmology prompts."""
    if not video.frames:
        return None

    with torch.no_grad():
        # Encode text prompts
        text_tokens = clip_tokenizer(COSMOS_PROMPTS).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Sample frames for relevance check
        sample_indices = np.linspace(0, len(video.frames) - 1, min(5, len(video.frames)), dtype=int)
        sample_frames = [video.frames[i] for i in sample_indices]

        # Encode frames
        images = torch.stack([
            clip_preprocess(Image.fromarray(frame)).to(device)
            for frame in sample_frames
        ])
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute max similarity across frames and prompts
        similarities = image_features @ text_features.T
        max_similarity = similarities.max().item()

    video.metadata["clip_similarity"] = max_similarity

    if max_similarity < threshold:
        return None

    return video


def compute_perceptual_hash(frame: np.ndarray) -> str:
    """Compute perceptual hash for a frame."""
    img = Image.fromarray(frame)
    return str(imagehash.phash(img))


def filter_duplicate(
    video: Video,
    seen_hashes: set[str],
    hash_threshold: int = 10,
) -> Video | None:
    """Filter duplicates using perceptual hashing."""
    if not video.frames:
        return None

    # Use middle frame for dedup
    mid_idx = len(video.frames) // 2
    phash = compute_perceptual_hash(video.frames[mid_idx])

    # Check similarity against seen hashes
    for seen in seen_hashes:
        if imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(seen) < hash_threshold:
            return None

    seen_hashes.add(phash)
    video.metadata["phash"] = phash

    return video
