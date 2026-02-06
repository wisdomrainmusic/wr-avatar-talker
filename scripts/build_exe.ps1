# Build EXE (ffmpeg bundled)
# Run from repo root:
# powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
  throw "Missing .venv. Run setup first."
}

.\.venv\Scripts\Activate.ps1

git submodule update --init --recursive

pyinstaller `
  --noconsole `
  --clean `
  --onedir `
  --name "WR-Avatar-Talker" `
  --add-binary "ffmpeg/ffmpeg.exe;." `
  ui\main.py

Write-Host "Build complete:"
Write-Host "dist\WR-Avatar-Talker\WR-Avatar-Talker.exe"
