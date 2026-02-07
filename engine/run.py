from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Callable, Optional


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def repo_root() -> Path:
    # engine/run.py -> parents[1] == repo root
    return Path(__file__).resolve().parents[1]


def app_root() -> Path:
    """
    Dev: repo root
    EXE (PyInstaller): sys._MEIPASS
    """
    if is_frozen() and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return repo_root()


def work_root() -> Path:
    """
    Dev: repo root
    EXE: folder where exe is located
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return repo_root()


def _valid_exe(p: Path) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size > 0
    except Exception:
        return False


def resolve_ffmpeg_path() -> str:
    """
    Resolve ffmpeg:
    - If UI set WR_AVATAR_TALKER_FFMPEG env -> use it
    - Dev: repo/ffmpeg/ffmpeg.exe
    - EXE: alongside exe / _internal layouts
    - Fallback: "ffmpeg"
    """
    env = os.environ.get("WR_AVATAR_TALKER_FFMPEG", "").strip()
    if env:
        p = Path(env)
        if _valid_exe(p) or env.lower() == "ffmpeg":
            return env

    if is_frozen():
        exe_dir = work_root()
        candidates = [
            exe_dir / "ffmpeg" / "ffmpeg.exe",
            exe_dir / "ffmpeg.exe",
            exe_dir / "_internal" / "ffmpeg" / "ffmpeg.exe",
            exe_dir / "_internal" / "ffmpeg.exe",
        ]
        for c in candidates:
            if _valid_exe(c):
                return str(c)

    if is_frozen() and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
        candidates = [
            base / "ffmpeg" / "ffmpeg.exe",
            base / "ffmpeg.exe",
            base / "_internal" / "ffmpeg" / "ffmpeg.exe",
            base / "_internal" / "ffmpeg.exe",
        ]
        for c in candidates:
            if _valid_exe(c):
                return str(c)

    local = repo_root() / "ffmpeg" / "ffmpeg.exe"
    if _valid_exe(local):
        return str(local)

    return "ffmpeg"


def _env_with_ffmpeg_on_path() -> dict:
    """Ensure SadTalker internal `ffmpeg` call works (it calls plain 'ffmpeg')."""
    env = os.environ.copy()
    ff = resolve_ffmpeg_path()
    try:
        p = Path(ff)
        if p.suffix.lower() == '.exe' and p.exists():
            ff_dir = str(p.parent)
            env['PATH'] = ff_dir + os.pathsep + env.get('PATH', '')
    except Exception:
        pass
    return env


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
    ffmpeg = resolve_ffmpeg_path()
    try:
        subprocess.check_call([ffmpeg, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        raise RuntimeError(f"ffmpeg not found. Tried: {ffmpeg}")


def postprocess_reels(input_mp4: Path, output_mp4: Path) -> None:
    ensure_dir(output_mp4.parent)
    # Keep original framing (no zoom) by fitting into 1080x1920 and padding.
    # Use a blurred background for a clean reels look.
    vf = (
        "split=2[bg][fg];"
        "[bg]scale=1080:1920,boxblur=20:1[bg2];"
        "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fg2];"
        "[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
    )
    run_cmd([
        resolve_ffmpeg_path(), "-y",
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


def postprocess_720p(input_mp4: Path, output_mp4: Path) -> None:
    """
    Standard 720p landscape for smaller file / faster playback.
    """
    ensure_dir(output_mp4.parent)
    # Preserve aspect ratio, pad to 1280x720 (no stretch)
    vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2"
    run_cmd([
        resolve_ffmpeg_path(), "-y",
        "-i", str(input_mp4),
        "-vf", vf,
        "-r", "30",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.1",
        "-crf", "20",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "160k",
        str(output_mp4)
    ])


def apply_cpu_thread_limits(cpu_threads: int) -> None:
    """
    Helps a lot on Windows CPU runs (MKL/OMP thread storms).
    Safe in dev + frozen.
    """
    t = max(1, int(cpu_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(t))
    os.environ.setdefault("MKL_NUM_THREADS", str(t))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(t))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(t))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(t))

    with suppress(Exception):
        import torch  # type: ignore
        torch.set_num_threads(t)
        torch.set_num_interop_threads(1)


def run_sadtalker_inference(inference_py: Path, argv_tail: list[str], sadtalker_dir: Path, *, patch_mode: str | None = None) -> None:
    """
    Dev (python/venv): run SadTalker from repo root so ./checkpoints resolves correctly.
    EXE (PyInstaller): keep cwd as SadTalker folder.
    """
    if not inference_py.exists():
        raise RuntimeError(f"SadTalker inference.py not found at: {inference_py}")

    old_cwd = os.getcwd()
    dev_repo_root = repo_root()

    try:
        if is_frozen():
            os.chdir(str(sadtalker_dir))
        else:
            os.chdir(str(dev_repo_root))

        if not is_frozen():
            # ✅ DEV: explicitly pass checkpoint_dir
            argv = [
                sys.executable,
                str(inference_py),
                "--checkpoint_dir", str(dev_repo_root / "checkpoints"),
                *argv_tail,
            ]
            p = subprocess.run(argv, text=True, capture_output=True, env=_env_with_ffmpeg_on_path())
            if p.returncode != 0:
                err = (p.stderr or '') + '\n' + (p.stdout or '')
                # Fallback: some SadTalker builds may not support '--still'
                if '--still' in argv and ('unrecognized arguments' in err or 'unknown argument' in err.lower()):
                    argv2 = [a for a in argv if a != '--still']
                    print('[WARN] SadTalker does not support --still; retrying without it.')
                    subprocess.check_call(argv2, env=_env_with_ffmpeg_on_path())
                    return
                raise subprocess.CalledProcessError(p.returncode, argv, output=p.stdout, stderr=p.stderr)
            return

        # Frozen: run in-process
        old_argv = sys.argv[:]
        old_path = sys.path[:]
        try:
            with suppress(Exception):
                if patch_mode:
                    os.environ.setdefault("WR_SADTALKER_PATCH_MODE", patch_mode)
                    from engine.sadtalker_patches import apply_patches  # type: ignore
                    apply_patches(patch_mode)

            if str(sadtalker_dir) not in sys.path:
                sys.path.insert(0, str(sadtalker_dir))

            sys.argv = [str(inference_py), *argv_tail]
            runpy.run_path(str(inference_py), run_name="__main__")
        finally:
            sys.argv = old_argv
            sys.path = old_path

    finally:
        os.chdir(old_cwd)


def generate(
    image: str | Path,
    audio: str | Path,
    preset: str = "calm",
    max_seconds: int = 60,
    reels: bool = False,
    out: str | Path = "output/out_reels.mp4",
    *,
    size: int = 256,
    preprocess: str = "crop",
    output_mode: str = "reels",
    cpu_threads: int = 4,
    keep_framing: bool = True,
    device_mode: str = "auto",  # auto|cpu|gpu
    eye_contact: str = "locked",  # locked|locked_plus|natural|expressive
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Path:
    def tick(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(int(pct), msg)

    tick(3, "Checking ffmpeg…")
    check_ffmpeg()
    apply_cpu_thread_limits(cpu_threads)

    image = Path(image)
    audio = Path(audio)
    out_path = Path(out)

    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio}")

    preprocess = (preprocess or "crop").strip().lower()
    if preprocess not in {"crop", "full"}:
        raise ValueError("preprocess must be 'crop' or 'full'")

    size = int(size)
    if size not in (256, 512):
        raise ValueError("size must be 256 or 512")

    output_mode = (output_mode or "reels").strip().lower()
    if output_mode not in {"reels", "720p", "none"}:
        raise ValueError("output_mode must be 'reels', '720p', or 'none'")

    device_mode = (device_mode or "auto").strip().lower()
    if device_mode not in {"auto", "cpu", "gpu"}:
        raise ValueError("device_mode must be 'auto', 'cpu', or 'gpu'")

    # Dev: repo/third_party/SadTalker
    # EXE: _MEIPASS/third_party/SadTalker (senin paketine göre)
    sadtalker_dir = app_root() / "third_party" / "SadTalker"
    inference_py = sadtalker_dir / "inference.py"
    if not inference_py.exists():
        raise RuntimeError(f"SadTalker not found at: {inference_py}")

    if preset == "calm":
        pose_style = "0"
        expression_scale = "0.45"
    elif preset == "normal":
        pose_style = "1"
        expression_scale = "0.65"
    elif preset == "ultra":
        # Experimental: fewer motion + faster render
        pose_style = "0"
        expression_scale = "0.35"
    else:
        raise ValueError("Invalid preset. Use 'calm', 'normal', or 'ultra'.")

    # Eye Contact tuning (YouTuber mode)
    eye_contact = (eye_contact or "locked").strip().lower()
    if eye_contact not in {"locked", "locked_plus", "natural", "expressive"}:
        raise ValueError("eye_contact must be 'locked', 'locked_plus', 'natural', or 'expressive'")

    try:
        expr_f = float(expression_scale)
    except Exception:
        expr_f = 0.45

    pose_i = 0
    with suppress(Exception):
        pose_i = int(pose_style)

    if eye_contact == "locked":
        # Keep head motion minimal so eyes stay on camera
        pose_i = 0
        expr_f = min(expr_f, 0.35)
    elif eye_contact == "locked_plus":
        # Locked+ (YouTuber Pro): ultra-stable head + micro expression boost
        pose_i = 0
        expr_f = 0.55
    elif eye_contact == "expressive":
        # Slightly more motion without going crazy
        pose_i = min(pose_i + 1, 3)
        expr_f = min(expr_f * 1.15, 0.80)

    pose_style = str(pose_i)
    expression_scale = f"{expr_f:.2f}"

    # Device selection
    dm = (device_mode or "auto").strip().lower()
    if dm not in {"auto", "cpu", "gpu"}:
        raise ValueError("device_mode must be 'auto', 'cpu', or 'gpu'")

    cuda_available = False
    with suppress(Exception):
        import torch  # type: ignore
        cuda_available = bool(torch.cuda.is_available())

    use_gpu = False
    if dm == "gpu":
        if not cuda_available:
            raise RuntimeError(
                "GPU mode requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build on this machine."
            )
        use_gpu = True
    elif dm == "cpu":
        use_gpu = False
    else:  # auto
        use_gpu = cuda_available

    ensure_dir(out_path.parent)
    tmp_dir = out_path.parent / "tmp_sadtalker"
    ensure_dir(tmp_dir)

    tick(12, "Trimming audio…")
    trimmed_audio = out_path.parent / "audio_trimmed.wav"
    run_cmd([
        resolve_ffmpeg_path(), "-y",
        "-i", str(audio),
        "-t", str(max_seconds),
        "-ac", "1",
        "-ar", "16000",
        str(trimmed_audio)
    ])

    tick(25, "Running SadTalker… (GPU)" if use_gpu else "Running SadTalker… (CPU)")

    argv_tail = [
        "--driven_audio", str(trimmed_audio),
        "--source_image", str(image),
        "--result_dir", str(tmp_dir),
        "--preprocess", preprocess,
        "--size", str(size),
        "--pose_style", pose_style,
        "--expression_scale", expression_scale,
        "--enhancer", "none",
        "--background_enhancer", "none",
    ]

    # Locked+ extras (non-breaking): forced still framing, optional blink ref, and in-process patches (frozen)
    patch_mode: str | None = None
    if eye_contact == "locked_plus":
        patch_mode = "locked_plus"

    # Device hint: inference.py uses CUDA automatically unless '--cpu' is given.
    if not use_gpu:
        argv_tail += ["--cpu"]

    # Keep original framing (avoid zoomed-in crop). This is SadTalker's --still.
    # Non-breaking: Locked+ always forces --still; other modes keep current behavior.
    if keep_framing or eye_contact == "locked_plus":
        argv_tail += ["--still"]

    # Optional blink reference (very low frequency) – only for Locked+ and only if file exists.
    if eye_contact == "locked_plus":
        blink_candidates = [
            work_root() / "assets" / "blink_ref.mp4",
            work_root() / "assets" / "blink_ref.mov",
            app_root() / "assets" / "blink_ref.mp4",
            app_root() / "assets" / "blink_ref.mov",
            app_root() / "third_party" / "SadTalker" / "examples" / "ref_eyeblink.mp4",
        ]
        for cand in blink_candidates:
            if cand.exists() and cand.is_file() and cand.stat().st_size > 0:
                argv_tail += ["--ref_eyeblink", str(cand)]
                break

    argv_tail += [
        "--input_yaw", "0", "0", "0",
        "--input_pitch", "0", "0", "0",
        "--input_roll", "0", "0", "0",
    ]
    run_sadtalker_inference(inference_py, argv_tail, sadtalker_dir, patch_mode=patch_mode)

    tick(78, "Collecting generated video…")
    generated = find_latest_mp4(tmp_dir)
    print(f"Generated video: {generated}")

    tick(88, "Writing final output…")
    # Back-compat: reels bool still works, but output_mode is now the main switch.
    if output_mode == "reels" or (reels and output_mode != "none"):
        postprocess_reels(generated, out_path)
        print(f"Reels output: {out_path}")
    elif output_mode == "720p":
        postprocess_720p(generated, out_path)
        print(f"720p output: {out_path}")
    else:
        if out_path.exists():
            out_path.unlink()
        out_path.write_bytes(generated.read_bytes())
        print(f"Output (no postprocess): {out_path}")

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Output was not created (or empty): {out_path}")

    tick(100, "Done.")
    return out_path


def main():
    parser = argparse.ArgumentParser("WR Avatar Talker (CPU / Reels)")
    parser.add_argument("--image", required=True, help="Input image path (jpg/png)")
    parser.add_argument("--audio", required=True, help="Input audio path (wav/mp3)")
    parser.add_argument("--preset", default="calm", choices=["calm", "normal", "ultra"], help="Motion preset")
    parser.add_argument("--max_seconds", type=int, default=60, help="Hard limit for duration (seconds)")
    parser.add_argument("--reels", action="store_true", help="Export as 1080x1920 reels mp4")
    parser.add_argument("--out", default="output/out_reels.mp4", help="Final output path (mp4)")
    parser.add_argument("--size", type=int, default=256, choices=[256, 512], help="SadTalker render size")
    parser.add_argument("--preprocess", default="crop", choices=["crop", "full"], help="SadTalker preprocess mode")
    parser.add_argument("--output_mode", default="reels", choices=["reels", "720p", "none"], help="Postprocess mode")
    parser.add_argument("--cpu_threads", type=int, default=4, help="CPU thread limit (OMP/MKL)")
    parser.add_argument("--device_mode", default="auto", choices=["auto", "cpu", "gpu"], help="Auto/force CPU/force GPU")
    parser.add_argument("--eye_contact", default="locked", choices=["locked", "locked_plus", "natural", "expressive"], help="Eye contact style")
    args = parser.parse_args()

    generate(
        image=args.image,
        audio=args.audio,
        preset=args.preset,
        max_seconds=args.max_seconds,
        reels=args.reels,
        out=args.out,
        size=args.size,
        preprocess=args.preprocess,
        output_mode=args.output_mode,
        cpu_threads=args.cpu_threads,
        device_mode=args.device_mode,
        eye_contact=args.eye_contact,
        progress_cb=None,
    )


if __name__ == "__main__":
    main()
