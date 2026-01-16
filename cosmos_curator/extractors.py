"""Frame, audio, and transcript extraction functions."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from cosmos_curator.pipeline import Video

logger = logging.getLogger(__name__)


def extract_frames(video: Video, fps: float = 1.0) -> Video:
    """Extract frames at specified FPS using ffmpeg."""
    if video.local_path is None:
        return video

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_pattern = Path(tmpdir) / "frame_%06d.png"
            cmd = [
                "ffmpeg",
                "-i", str(video.local_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                str(output_pattern),
                "-y",
                "-loglevel", "error",
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            frames = []
            for frame_path in sorted(Path(tmpdir).glob("frame_*.png")):
                img = Image.open(frame_path).convert("RGB")
                frames.append(np.array(img))

            video.frames = frames
    except subprocess.CalledProcessError as e:
        logger.warning(f"Frame extraction failed for {video.url}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error in frame extraction: {e}")

    return video


def extract_audio(video: Video, sample_rate: int = 16000) -> Video:
    """Extract audio as 16kHz mono using ffmpeg."""
    if video.local_path is None:
        return video

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg",
            "-i", str(video.local_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            tmp_path,
            "-y",
            "-loglevel", "error",
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            with wave.open(tmp_path, "rb") as wf:
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                video.audio = audio
    except Exception as e:
        logger.warning(f"Audio extraction failed for {video.url}: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return video


def transcribe(video: Video, model_size: str = "base") -> Video:
    """Transcribe audio using faster-whisper."""
    if video.audio is None or len(video.audio) == 0:
        return video

    tmp_path = None
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="auto", compute_type="auto")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                audio_int16 = (video.audio * 32768).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())

        segments, info = model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        video.transcript = transcript.strip()
        video.metadata["language"] = info.language
        video.metadata["language_probability"] = info.language_probability
    except Exception as e:
        logger.warning(f"Transcription failed for {video.url}: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    return video


def embed(
    video: Video,
    clip_model: torch.nn.Module,
    clip_preprocess,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Video:
    """Generate CLIP embeddings for video frames."""
    if not video.frames:
        return video

    try:
        with torch.no_grad():
            images = torch.stack([
                clip_preprocess(Image.fromarray(frame)).to(device)
                for frame in video.frames
            ])
            embeddings = clip_model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            video.embeddings = embeddings.cpu().numpy()
    except Exception as e:
        logger.warning(f"Embedding generation failed for {video.url}: {e}")

    return video


def classify(video: Video, taxonomy: dict[str, list[str]] | None = None) -> Video:
    """Classify video based on embeddings and taxonomy."""
    if video.embeddings is None:
        return video

    if taxonomy is None:
        taxonomy = {
            "galaxy": ["spiral galaxy", "elliptical galaxy", "galaxy cluster"],
            "nebula": ["planetary nebula", "emission nebula", "dark nebula"],
            "stellar": ["star formation", "supernova", "neutron star", "pulsar"],
            "cosmology": ["big bang", "cosmic microwave background", "dark matter"],
            "solar_system": ["planet", "asteroid", "comet", "moon"],
            "instrumentation": ["telescope", "observatory", "satellite"],
        }

    video.labels["taxonomy"] = taxonomy
    video.labels["categories"] = []

    mean_embedding = video.embeddings.mean(axis=0)
    video.labels["mean_embedding_norm"] = float(np.linalg.norm(mean_embedding))

    return video
