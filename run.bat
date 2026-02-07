@echo off
setlocal
cd /d "%~dp0"

call venv\Scripts\activate.bat

REM Repo root'u sys.path'e garanti eklemek için:
set PYTHONPATH=%cd%

REM (opsiyonel) ffmpeg'i sabitlemek istersen:
REM set WR_AVATAR_TALKER_FFMPEG=%cd%\ffmpeg\ffmpeg.exe

python ui\main.py
pause
