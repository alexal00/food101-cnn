"""Food-101 preprocessing and augmentation transforms."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image
from torchvision import transforms

from food101_cnn.data.image_ops import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PAD_FILL,
    prepare_pil_image,
    resize_with_padding,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PadResize:
    """Resize an image with aspect-ratio preserving padding."""

    size: int = DEFAULT_IMAGE_SIZE
    fill: tuple[int, int, int] = DEFAULT_PAD_FILL

    def __call__(self, image: Image.Image) -> Image.Image:
        return resize_with_padding(image, size=self.size, fill=self.fill)


def build_train_transform(
    config: Mapping[str, Any] | None = None,
    *,
    image_size: int | None = None,
) -> transforms.Compose:
    """Build conservative training transforms."""
    config = config or {}
    augmentation = config.get("augmentation", {})
    size = _image_size(config, image_size)

    transform_steps: list[object] = [PadResize(size=size)]

    horizontal_flip_p = float(augmentation.get("horizontal_flip_p", 0.5))
    if horizontal_flip_p > 0:
        transform_steps.append(transforms.RandomHorizontalFlip(p=horizontal_flip_p))

    color_jitter = augmentation.get("color_jitter", {})
    if isinstance(color_jitter, Mapping) and color_jitter:
        transform_steps.append(
            transforms.ColorJitter(
                brightness=float(color_jitter.get("brightness", 0.0)),
                contrast=float(color_jitter.get("contrast", 0.0)),
                saturation=float(color_jitter.get("saturation", 0.0)),
                hue=float(color_jitter.get("hue", 0.0)),
            )
        )

    rotation_degrees = float(augmentation.get("rotation_degrees", 0.0))
    if rotation_degrees > 0:
        transform_steps.append(
            transforms.RandomRotation(
                degrees=rotation_degrees,
                fill=DEFAULT_PAD_FILL,
            )
        )

    transform_steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(transform_steps)


def build_eval_transform(
    config: Mapping[str, Any] | None = None,
    *,
    image_size: int | None = None,
) -> transforms.Compose:
    """Build deterministic evaluation and inference transforms."""
    size = _image_size(config or {}, image_size)
    return transforms.Compose(
        [
            PadResize(size=size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


get_train_transform = build_train_transform
get_eval_transform = build_eval_transform


def _image_size(config: Mapping[str, Any], override: int | None) -> int:
    if override is not None:
        return override

    data = config.get("data", {})
    if isinstance(data, Mapping) and "image_size" in data:
        return int(data["image_size"])

    return DEFAULT_IMAGE_SIZE
