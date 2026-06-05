"""PIL-only image preprocessing operations."""

from PIL import Image, ImageOps

DEFAULT_IMAGE_SIZE = 224
DEFAULT_PAD_FILL = (0, 0, 0)


def prepare_pil_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction and convert to RGB."""
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_with_padding(
    image: Image.Image,
    *,
    size: int = DEFAULT_IMAGE_SIZE,
    fill: tuple[int, int, int] = DEFAULT_PAD_FILL,
) -> Image.Image:
    """Resize a PIL image to ``size`` x ``size`` without cropping."""
    if size <= 0:
        raise ValueError("Image size must be positive.")

    rgb_image = prepare_pil_image(image)
    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = rgb_image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.BICUBIC,
    )

    canvas = Image.new("RGB", (size, size), color=fill)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas
