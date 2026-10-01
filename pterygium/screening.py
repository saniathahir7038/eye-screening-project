"""Shared validation and inference functions for the pterygium screening interface."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from prepare_dataset import preprocess_image


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "phase4a" / "corrected_model.keras"
DEFAULT_SUMMARY_PATH = ROOT / "artifacts" / "phase4a" / "phase4a_summary.json"
NO_VISIBLE_PATTERN = "No visible pterygium pattern"
SUSPECTED_PTERYGIUM = "Suspected pterygium"


class ImageInputError(ValueError):
    """Raised when an upload cannot be safely decoded as a supported image."""


@dataclass(frozen=True)
class QualityConfig:
    min_width: int = 224
    min_height: int = 224
    min_aspect_ratio: float = 0.75
    max_aspect_ratio: float = 2.0
    min_brightness: float = 45.0
    max_brightness: float = 210.0
    min_contrast: float = 18.0
    min_sharpness: float = 20.0


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    width: int
    height: int
    aspect_ratio: float
    brightness: float
    contrast: float
    sharpness: float
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ScreeningRuntime:
    model: object
    gradcam_model: object
    threshold: float
    model_path: str


@dataclass(frozen=True)
class ScreeningResult:
    label: str
    suspected: bool
    model_score: float
    threshold: float
    processed_image: Image.Image
    heatmap_overlay: Image.Image


def decode_uploaded_image(data: bytes, maximum_bytes: int = 15 * 1024 * 1024) -> Image.Image:
    if not data:
        raise ImageInputError("The uploaded file is empty.")
    if len(data) > maximum_bytes:
        raise ImageInputError("The image is larger than the 15 MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as candidate:
            candidate.verify()
        with Image.open(BytesIO(data)) as candidate:
            candidate.load()
            image = ImageOps.exif_transpose(candidate).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImageInputError("The file is not a readable PNG or JPEG image.") from error
    return image


def quality_metrics(image: Image.Image) -> dict[str, float]:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    resized = np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    return {
        "width": int(rgb.width),
        "height": int(rgb.height),
        "aspect_ratio": float(rgb.width / rgb.height),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def assess_image_quality(image: Image.Image, config: QualityConfig | None = None) -> QualityResult:
    config = config or QualityConfig()
    metrics = quality_metrics(image)
    issues = []
    if metrics["width"] < config.min_width or metrics["height"] < config.min_height:
        issues.append(f"Use an image of at least {config.min_width} × {config.min_height} pixels.")
    if not config.min_aspect_ratio <= metrics["aspect_ratio"] <= config.max_aspect_ratio:
        issues.append("Retake a close-up landscape or near-square eye photograph.")
    if metrics["brightness"] < config.min_brightness:
        issues.append("The image is too dark. Use brighter, even lighting.")
    elif metrics["brightness"] > config.max_brightness:
        issues.append("The image is overexposed. Reduce glare and direct light.")
    if metrics["contrast"] < config.min_contrast:
        issues.append("The image has too little contrast. Improve lighting and camera focus.")
    if metrics["sharpness"] < config.min_sharpness:
        issues.append("The image appears blurred. Hold the camera steady and refocus.")
    return QualityResult(
        accepted=not issues,
        issues=tuple(issues),
        **metrics,
    )


def load_runtime(model_path: Path = DEFAULT_MODEL_PATH,
                 summary_path: Path = DEFAULT_SUMMARY_PATH) -> ScreeningRuntime:
    model_path, summary_path = Path(model_path), Path(summary_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Corrected model not found: {model_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Phase 4A summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "accepted" or summary.get("test_set_inference_performed") is not False:
        raise ValueError("Phase 4A model has not passed the required validation-only acceptance gate.")
    threshold = float(summary["corrected"]["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("The Phase 4A threshold is invalid.")

    import tensorflow as tf
    from evaluate import build_gradcam_model

    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3) or tuple(model.output_shape[1:]) != (1,):
        raise ValueError(f"Unexpected corrected model shape: {model.input_shape} -> {model.output_shape}")
    return ScreeningRuntime(
        model=model,
        gradcam_model=build_gradcam_model(model),
        threshold=threshold,
        model_path=str(model_path.resolve()),
    )


def create_heatmap_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if heatmap.shape != base.shape[:2] or not np.isfinite(heatmap).all():
        raise ValueError("Grad-CAM heatmap is invalid or has the wrong dimensions.")
    coloured = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(base, 0.58, coloured, 0.42, 0)
    return Image.fromarray(blended)


def run_screening(runtime: ScreeningRuntime, image: Image.Image) -> ScreeningResult:
    from evaluate import gradcam

    processed, _ = preprocess_image(ImageOps.exif_transpose(image).convert("RGB"), 224, 0.20)
    heatmap, score = gradcam(runtime.gradcam_model, np.asarray(processed, dtype=np.float32))
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("The model returned an invalid score.")
    suspected = score >= runtime.threshold
    return ScreeningResult(
        label=SUSPECTED_PTERYGIUM if suspected else NO_VISIBLE_PATTERN,
        suspected=suspected,
        model_score=score,
        threshold=runtime.threshold,
        processed_image=processed,
        heatmap_overlay=create_heatmap_overlay(processed, heatmap),
    )


def screen_file(path: Path, model_path: Path = DEFAULT_MODEL_PATH,
                summary_path: Path = DEFAULT_SUMMARY_PATH) -> dict:
    path = Path(path)
    image = decode_uploaded_image(path.read_bytes())
    quality = assess_image_quality(image)
    output = {"image": str(path), "quality": asdict(quality)}
    if not quality.accepted:
        output["result"] = "Image quality insufficient — retake photograph"
        return output
    result = run_screening(load_runtime(model_path, summary_path), image)
    output["result"] = {
        "label": result.label,
        "model_score": result.model_score,
        "threshold": result.threshold,
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="PNG or JPEG eye image")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()
    print(json.dumps(screen_file(args.image, args.model, args.summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
