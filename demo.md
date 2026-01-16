# Demo: Cosmos Curator in Action

## Setup

```bash
# Install dependencies
pip install -e .

# Verify ffmpeg is available
ffmpeg -version
```

## Quick Test (3 videos)

Create `sample_urls.txt`:
```
https://www.youtube.com/watch?v=GoW8Tf7hTGA
https://www.youtube.com/watch?v=XBr4GkRnY04
https://www.youtube.com/watch?v=xS89EolXxio
```

Run the pipeline:
```bash
python run.py --urls sample_urls.txt --output ./demo_output --min-dur 5
```

## Inspect Output

```bash
# List generated shards
ls -lh demo_output/

# Peek inside a shard
tar -tvf demo_output/shard-000000.tar | head -20

# Extract and view metadata
tar -xf demo_output/shard-000000.tar -C /tmp/
cat /tmp/000000.json | python -m json.tool
```

## Load with WebDataset

```python
import webdataset as wds

dataset = wds.WebDataset("demo_output/shard-*.tar")

for sample in dataset:
    print(f"Key: {sample['__key__']}")
    print(f"Transcript: {sample['txt'].decode()[:200]}...")
    break
```

## Config File Option

Create `config.yaml`:
```yaml
input_urls: sample_urls.txt
output_dir: ./demo_output
min_resolution: 480
min_duration: 5
max_duration: 600
clip_threshold: 0.2
```

Run:
```bash
python run.py --config config.yaml
```

## Expected Output

```
demo_output/
└── shard-000000.tar
    ├── 000000.mp4      # 15MB video
    ├── 000000.json     # {"url": "...", "metadata": {...}}
    ├── 000000.txt      # "In this visualization of the cosmos..."
    └── 000000.npy      # (N, 512) CLIP embeddings
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No videos pass filter | Lower `--clip-thresh` to 0.15 |
| CUDA out of memory | Process fewer videos or use CPU |
| Download fails | Check yt-dlp version: `pip install -U yt-dlp` |
