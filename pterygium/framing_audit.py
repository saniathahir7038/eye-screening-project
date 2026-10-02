"""Mirror the Android lower-margin framing heuristic for offline auditing.

This is a framing experiment, not a clinical image-quality or disease detector.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from framing import suggest_bottom_crop


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted((ROOT / "artifacts/phase2/processed_images").glob("*.png"))
    cropped = []
    for path in paths:
        with Image.open(path) as image:
            bottom = suggest_bottom_crop(image)
        if bottom < 1:
            cropped.append((path.name, bottom))
    print(f"Images: {len(paths)}; suggested crop: {len(cropped)}")
    for row in cropped[:40]:
        print(*row)


if __name__ == "__main__":
    main()
