"""
WR Avatar Talker - Engine entry point
MVP skeleton
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="WR Avatar Talker")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--audio", required=True, help="Path to input audio (wav/mp3)")
    parser.add_argument("--preset", default="calm", help="Motion preset")

    args = parser.parse_args()

    image_path = Path(args.image)
    audio_path = Path(args.audio)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print("WR Avatar Talker")
    print(f"Image : {image_path}")
    print(f"Audio : {audio_path}")
    print(f"Preset: {args.preset}")
    print("Status : Engine skeleton ready")


if __name__ == "__main__":
    main()
