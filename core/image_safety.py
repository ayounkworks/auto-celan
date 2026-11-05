"""Dependency-free validation for downloaded image resource limits."""


def validate_byte_size(size: int, max_bytes: int, label: str = "download") -> int:
    if size < 0:
        raise ValueError(f"{label} size cannot be negative")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    if size > max_bytes:
        raise ValueError(
            f"{label} exceeds configured limit: {size:,} bytes > {max_bytes:,} bytes"
        )
    return size


def validate_source_pixels(width: int, height: int, max_pixels: int) -> int:
    if width < 1 or height < 1:
        raise ValueError(f"Invalid source image dimensions: {width}x{height}")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    pixels = width * height
    if pixels > max_pixels:
        raise ValueError(
            f"Source image exceeds pixel limit: {width}x{height}={pixels:,} "
            f"> {max_pixels:,} pixels"
        )
    return pixels
