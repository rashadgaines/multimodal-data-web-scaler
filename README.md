# Multimodal Data Web Scaler

Scalable pipeline for curating video datasets from web sources.

## How It Works

```
Input URLs
    ↓
Download (yt-dlp)
    ↓
Technical Filter (resolution ≥720p, duration 10s-30min)
    ↓
Frame Extraction (ffmpeg @ configurable FPS)
    ↓
Relevance Filter (CLIP zero-shot similarity)
    ↓
Duplicate Filter (perceptual hashing)
    ↓
Audio Extraction (16kHz mono)
    ↓
Transcription (faster-whisper)
    ↓
Embedding Generation (CLIP)
    ↓
WebDataset Shards (.tar)
```

## Output

Each shard contains:
- `.mp4` - video file
- `.json` - metadata (title, uploader, timestamps, CLIP scores)
- `.txt` - transcript
- `.npy` - frame embeddings

## Install

```bash
# Prerequisites: Python 3.11+, ffmpeg
pip install -e .
```

## Usage

```bash
# Basic
python run.py --urls urls.txt --output ./output

# With config
python run.py --config config.yaml

# Distributed (Ray cluster)
python run.py --urls urls.txt --output ./output --ray auto
```

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--min-res` | 720 | Minimum height (pixels) |
| `--min-dur` | 10 | Minimum duration (seconds) |
| `--max-dur` | 1800 | Maximum duration (seconds) |
| `--clip-thresh` | 0.25 | CLIP similarity threshold |
| `--verbose` | false | Debug logging |

## License

MIT
