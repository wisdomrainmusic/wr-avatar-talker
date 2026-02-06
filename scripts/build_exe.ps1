# Build EXE (folder distribution) for WR Avatar Talker
# Run from repo root:
# powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
  throw "Missing .venv. Run setup first."
}

.\.venv\Scripts\Activate.ps1

# Ensure submodule exists
git submodule update --init --recursive

# Build using PyInstaller (onedir distribution)
pyinstaller `
  --noconsole `
  --name "WR-Avatar-Talker" `
  --clean `
  --onedir `
  ui\main.py

Write-Host "Build complete: dist\WR-Avatar-Talker\WR-Avatar-Talker.exe"
Write-Host "Next: copy ffmpeg\ffmpeg.exe and ensure models\ is placed next to exe."
