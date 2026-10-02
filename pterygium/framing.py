"""Conservative automatic framing shared by the Streamlit and Android prototypes.

The Android implementation mirrors these constants and calculations. Framing is a
technical adjustment, not a calibrated disease prediction or eye detector.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class FramingResult:
    image: Image.Image
    removed_bottom_fraction: float

    @property
    def adjusted(self) -> bool:
        return self.removed_bottom_fraction > 0


def suggest_bottom_crop(image: Image.Image) -> float:
    """Return the retained height while leaving the full width and top untouched."""
    preview = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    rgb = np.asarray(preview, dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    vertical_edges = np.abs(gray[2:, 13:115] - gray[:-2, 13:115]).mean(axis=1)
    energy = np.pad(vertical_edges, (1, 1), mode="edge")
    energy = np.convolve(np.pad(energy, (4, 4), mode="edge"),
                         np.ones(9, dtype=np.float64) / 9, mode="valid")
    sorted_energy = np.sort(energy)
    baseline = float(sorted_energy[int(0.20 * 127)])
    peak = float(sorted_energy[int(0.90 * 127)])
    if peak < baseline + 1.5:
        return 1.0
    activity = np.maximum(0, energy - baseline)
    if activity.sum() <= 0:
        return 1.0
    last_active = int(np.searchsorted(np.cumsum(activity), 0.95 * activity.sum()))
    bottom = min(1.0, (last_active + 1) / 128 + 0.12)
    if bottom < 0.75 or 1 - bottom < 0.12:
        return 1.0
    lower = rgb[112:128, 13:115]
    if float((lower[:, :, 0] - lower[:, :, 1]).mean()) < 48:
        return 1.0
    if float((lower[:, :, 1] - lower[:, :, 2]).mean()) < 15:
        return 1.0
    return bottom


def auto_frame(image: Image.Image) -> FramingResult:
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    fraction = suggest_bottom_crop(oriented)
    height = max(1, min(oriented.height, math.floor(oriented.height * fraction + 0.5)))
    if height == oriented.height:
        return FramingResult(oriented, 0.0)
    return FramingResult(oriented.crop((0, 0, oriented.width, height)),
                         1 - height / oriented.height)
