#!/usr/bin/env python
"""Export the shared SmartKC/DEDector mire segmentor for PyTorch Android Lite."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from get_center.network_unet import model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("get_center/segment_and_get_center_epoch_557_iter_14.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "../pterygium/android_app/app/src/main/assets/"
            "placido_segmentation.ptl"
        ),
    )
    parser.add_argument(
        "--validation-image",
        type=Path,
        default=Path("data/sample_images/kc_left.jpg"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    network = model()
    network.load_state_dict(checkpoint)
    network.eval()

    example = torch.zeros((1, 3, 512, 512), dtype=torch.float32)
    traced = torch.jit.trace(network, example)
    mobile = optimize_for_mobile(traced)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mobile._save_for_lite_interpreter(str(args.output))

    transform = Compose(
        [
            Resize((512, 512)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = transform(Image.open(args.validation_image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        reference = network(image)
        candidate = traced(image)
    max_difference = float(torch.max(torch.abs(reference - candidate)))
    label_agreement = float(
        np.mean(
            torch.argmax(reference, dim=1).numpy()
            == torch.argmax(candidate, dim=1).numpy()
        )
    )

    print(f"output={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"max_output_difference={max_difference:.8f}")
    print(f"argmax_label_agreement={label_agreement:.8f}")


if __name__ == "__main__":
    main()
