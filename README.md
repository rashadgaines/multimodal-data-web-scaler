# Cosmos Curator

A scalable pipeline for curating cosmology and astronomy video datasets.

## Overview

Cosmos Curator downloads videos, filters by quality and relevance, extracts frames and audio, transcribes speech, generates CLIP embeddings, and outputs ML-ready WebDataset shards.

```
URLs → Download → Filter → Extract → Embed → WebDataset
```

## Installation

### Prerequisites

- Python 3.11+
- FFmpeg (`brew install ffmpeg` or `apt install ffmpeg`)
- CUDA (optional, for GPU acceleration)

### Install

```bash
git clone https://github.com/yourusername/cosmos-curator.git
cd cosmos-curator
pip install -e .
```

## Quick Start

```bash
# Create a file with video URLs
echo "https://www.youtube.com/watch?v=VIDEO_ID" > urls.txt

# Run the pipeline
python run.py --urls urls.txt --output ./output

# With verbose logging
python run.py --urls urls.txt --output ./output --verbose
```

## Configuration

### CLI Options

```
--urls, -u          Path to file with video URLs
--output, -o        Output directory (default: ./output)
--config, -c        Path to YAML config file
--min-res           Minimum resolution in pixels (default: 720)
--min-dur           Minimum duration in seconds (default: 10)
--max-dur           Maximum duration in seconds (default: 1800)
--clip-thresh       CLIP similarity threshold (default: 0.25)
--ray               Ray cluster address
--verbose, -v       Enable debug logging
```

### YAML Config

```yaml
input_urls: urls.txt
output_dir: ./output
min_resolution: 720
min_duration: 10
max_duration: 1800
clip_threshold: 0.25
frame_fps: 1.0
whisper_model: base
clip_model: ViT-B-32
clip_pretrained: laion2b_s34b_b79k
```

```bash
python run.py --config config.yaml
```

## Pipeline Stages

| Stage | Description |
|-------|-------------|
| **Download** | Fetch videos via yt-dlp with metadata |
| **Technical Filter** | Reject videos below resolution/duration thresholds |
| **Relevance Filter** | CLIP zero-shot classification against cosmology prompts |
| **Duplicate Filter** | Perceptual hashing to remove near-duplicates |
| **Frame Extraction** | Sample frames at configurable FPS |
| **Audio Extraction** | Extract 16kHz mono audio |
| **Transcription** | Speech-to-text via faster-whisper |
| **Embedding** | Generate CLIP embeddings for frames |
| **Output** | Write WebDataset tar shards |

## Output Format

```
output/
└── shard-000000.tar
    ├── 000000.mp4      # Video file
    ├── 000000.json     # Metadata and labels
    ├── 000000.txt      # Transcript
    └── 000000.npy      # CLIP embeddings (N, 512)
```

### Loading Output

```python
import webdataset as wds
import json

dataset = wds.WebDataset("output/shard-*.tar")

for sample in dataset:
    metadata = json.loads(sample["json"])
    transcript = sample["txt"].decode()
    print(f"Title: {metadata['metadata']['title']}")
    print(f"Transcript: {transcript[:100]}...")
```

## Distributed Processing

Use Ray for distributed processing across multiple machines:

```bash
# Start Ray cluster
ray start --head

# Run with Ray
python run.py --urls urls.txt --output ./output --ray auto
```

## Project Structure

```
cosmos_curator/
├── __init__.py      # Package exports
├── config.py        # Pydantic configuration model
├── filters.py       # Quality and relevance filters
├── extractors.py    # Frame/audio/transcript extraction
└── pipeline.py      # Core pipeline and Video dataclass
```

## Requirements

- ray[data]>=2.40
- webdataset>=0.2
- yt-dlp>=2024.1
- torch>=2.5
- open-clip-torch>=2.28
- faster-whisper>=1.1
- pydantic-settings>=2.6

## License

MIT License - see [LICENSE](LICENSE) for details.
