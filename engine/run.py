import argparse
import subprocess
from pathlib import Path
import glob
import os
import sys


def run_cmd(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd) + "\n")
    subprocess.check_call(cmd)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_latest_mp4(folder: Path) -> Path:
    mp4s = list(folder.rglob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 found under: {folder}")
    mp4s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0]


def check_ffmpeg() -> None:
    try:
        subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or add it to PATH.")


def postprocess_reels(input_mp4: Path, output_mp4: Path) -> None:
    """
    Reels-safe export:
    - 1080x1920 (9:16)
    - 30 fps
    - H.264 yuv420p + AAC
    """
    ensure_dir(output_mp4.parent)

    # Make sure it fills 9:16 without black bars:
    # 1) scale to height 1920
    # 2) center crop width 1080
    vf = "scale=-2:1920,crop=1080:1920"

    run_cmd([
        "ffmpeg", "-y",
        "-i", str(input_mp4),
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_mp4)
    ])


def main():
    parser = argparse.ArgumentParser("WR Avatar Talker (CPU / Reels)")
    parser.add_argument("--image", required=True, help="Input image path (jpg/png)")
    parser.add_argument("--audio", required=True, help="Input audio path (wav/mp3)")
    parser.add_argument("--preset", default="calm", choices=["calm", "normal"], help="Motion preset")
    parser.add_argument("--max_seconds", type=int, default=60, help="Hard limit for duration (seconds)")
    parser.add_argument("--reels", action="store_true", help="Export as 1080x1920 reels mp4")
    parser.add_argument("--out", default="output/out_reels.mp4", help="Final output path (mp4)")
    args = parser.parse_args()

    check_ffmpeg()

    image = Path(args.image)
    audio = Path(args.audio)
    out_path = Path(args.out)

    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    repo_root = Path(__file__).resolve().parents[1]
    sadtalker_dir = repo_root / "third_party" / "SadTalker"
    inference_py = sadtalker_dir / "inference.py"

    if not inference_py.exists():
        raise RuntimeError("SadTalker not found. Run: git submodule update --init --recursive")

    # ----- Preset tuning: minimal head motion, Reels avatar friendly -----
    # calm: more stable face, minimal expression
    if args.preset == "calm":
        pose_style = "0"
        expression_scale = "0.45"
    else:
        pose_style = "1"
        expression_scale = "0.65"

    # Output workspace
    ensure_dir(out_path.parent)
    tmp_dir = out_path.parent / "tmp_sadtalker"
    ensure_dir(tmp_dir)

    # 1) Trim audio to max_seconds + convert to 16k mono wav (SadTalker-friendly)
    trimmed_audio = out_path.parent / "audio_trimmed.wav"
    run_cmd([
        "ffmpeg", "-y",
        "-i", str(audio),
        "-t", str(args.max_seconds),
        "-ac", "1",
        "-ar", "16000",
        str(trimmed_audio)
    ])

    # 2) Run SadTalker inference
    # Notes:
    # - --size 512 is critical for CPU speed
    # - enhancers off for speed/naturalness
    # - yaw/pitch/roll forced to zero to avoid turning head
    cmd = [
        sys.executable, str(inference_py),
        "--driven_audio", str(trimmed_audio),
        "--source_image", str(image),
        "--result_dir", str(tmp_dir),
        "--preprocess", "full",
        "--size", "512",
        "--pose_style", pose_style,
        "--expression_scale", expression_scale,
        "--enhancer", "none",
        "--background_enhancer", "none",
        "--input_yaw", "0", "0", "0",
        "--input_pitch", "0", "0", "0",
        "--input_roll", "0", "0", "0",
    ]

    run_cmd(cmd)

    # 3) Find SadTalker result mp4 (latest)
    generated = find_latest_mp4(tmp_dir)
    print(f"Generated video: {generated}")

    # 4) Reels export (optional)
    if args.reels:
        postprocess_reels(generated, out_path)
        print(f"Reels output: {out_path}")
    else:
        # If not reels, copy to out_path
        ensure_dir(out_path.parent)
        if out_path.exists():
            out_path.unlink()
        out_path.write_bytes(generated.read_bytes())
        print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
