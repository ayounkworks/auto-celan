import pytest

from core.image_safety import validate_byte_size, validate_source_pixels


def test_download_size_within_limit_is_accepted():
    assert validate_byte_size(1024, 2048) == 1024


def test_download_size_above_limit_is_rejected():
    with pytest.raises(ValueError, match="exceeds configured limit"):
        validate_byte_size(2049, 2048, "Drive file")


def test_negative_download_size_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_byte_size(-1, 2048)


def test_invalid_download_limit_is_rejected():
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        validate_byte_size(1, 0)


def test_source_pixel_count_is_accepted_within_limit():
    assert validate_source_pixels(1024, 10000, 20_000_000) == 10_240_000


def test_source_pixel_count_is_rejected_before_decode():
    with pytest.raises(ValueError, match="exceeds pixel limit"):
        validate_source_pixels(1024, 50000, 50_000_000)


@pytest.mark.parametrize(("width", "height"), [(0, 1), (1, 0), (-1, 2)])
def test_invalid_source_dimensions_are_rejected(width, height):
    with pytest.raises(ValueError, match="Invalid source image dimensions"):
        validate_source_pixels(width, height, 50_000_000)
