# WR Avatar Talker

Offline Windows tool to generate talking avatar videos from a single image and audio file.

## Features
- Single photo + audio → MP4 video
- Lip-sync with subtle head motion
- Optimized for short-form content (Reels, Shorts)
- Fully offline (audio generated externally, e.g. ElevenLabs)

## Planned Stack
- Python 3.10+
- PyTorch
- SadTalker (core engine)
- FFmpeg

## Usage (MVP – CLI)
```bash
python engine/run.py --image photo.jpg --audio voice.wav --preset calm

Presets

calm (minimal head motion, avatar-friendly)

normal (slightly more expressive)

Status

🚧 Initial setup – engine skeleton in progress.
```
