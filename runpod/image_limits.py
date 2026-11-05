"""Safe image dimension planning for RunPod inference."""

import math


def validate_decode_pixels(width: int, height: int, max_decode_pixels: int) -> int:
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if max_decode_pixels < 1:
        raise ValueError("max_decode_pixels must be positive")
    pixels = width * height
    if pixels > max_decode_pixels:
        raise ValueError(
            f"Image too large to decode safely: {width}x{height}={pixels:,} pixels; "
            f"decode limit is {max_decode_pixels:,}"
        )
    return pixels


def fit_dimensions(
    width: int,
    height: int,
    max_height: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Scale down proportionally to satisfy both height and total-pixel limits."""
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")
    if max_height < 1 or max_pixels < 1:
        raise ValueError("Image limits must be positive")

    scale = min(
        1.0,
        max_height / height,
        math.sqrt(max_pixels / (width * height)),
    )
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))

    # Correct possible rounding overflow while preserving aspect ratio as closely
    # as possible. This should only adjust a dimension by a pixel.
    while target_width * target_height > max_pixels:
        if target_width / width >= target_height / height and target_width > 1:
            target_width -= 1
        elif target_height > 1:
            target_height -= 1
        else:
            break
    if target_height > max_height:
        target_height = max_height
    return target_width, target_height
