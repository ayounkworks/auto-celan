"""Local input discovery and output path helpers."""
from pathlib import Path
import re

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def default_output_dir(input_dir: Path) -> Path:
    """Place the cleaned folder beside the source folder."""
    return input_dir.with_name(f"{input_dir.name}_cleaned")


def discover_images(input_dir: Path) -> list[Path]:
    """Return supported files in natural filename order (non-recursive)."""
    def natural_key(path: Path):
        return [
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ]

    return sorted(
        (
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in VALID_IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
