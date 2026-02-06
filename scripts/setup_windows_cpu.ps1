# CPU-only setup (Windows)
# Run:
# powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows_cpu.ps1

$ErrorActionPreference = "Stop"

Write-Host "== WR Avatar Talker | CPU Setup =="

python --version

if (!(Test-Path ".venv")) {
  python -m venv .venv
}

.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel setuptools

# CPU PyTorch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Project deps
python -m pip install -r requirements.txt

# Submodule init
git submodule update --init --recursive

Write-Host "== Done. Next: python engine\run.py --image ... --audio ... --preset calm --reels =="
