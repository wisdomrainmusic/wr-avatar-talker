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

## GUI (Windows)
Run:
```bash
python ui/main.py
```

Build EXE (Windows)

Setup venv + deps (CPU):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cpu.ps1
```

Build:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1
```

Output:
```
dist/WR-Avatar-Talker/WR-Avatar-Talker.exe
```

Notes:
- Place ffmpeg/ffmpeg.exe next to the exe (or ensure ffmpeg is in PATH)
- Place models/ folder next to the exe (local-only, not committed)
