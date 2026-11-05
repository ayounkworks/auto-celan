import pytest

from runpod.image_limits import fit_dimensions, validate_decode_pixels


@pytest.mark.parametrize(
    ("width", "height", "max_height", "max_pixels"),
    [
        (1024, 30000, 20000, 25_000_000),
        (3000, 10000, 20000, 25_000_000),
        (1024, 10000, 20000, 25_000_000),
        (800, 1200, 20000, 25_000_000),
    ],
)
def test_fit_dimensions_respects_both_limits(
    width, height, max_height, max_pixels
):
    target_width, target_height = fit_dimensions(
        width, height, max_height, max_pixels
    )
    assert target_width >= 1
    assert target_height >= 1
    assert target_height <= max_height
    assert target_width * target_height <= max_pixels
    assert target_width <= width
    assert target_height <= height


def test_tall_webtoon_page_is_resized_instead_of_rejected_by_inference_limit():
    target = fit_dimensions(1024, 30000, max_height=20000, max_pixels=25_000_000)
    assert target != (1024, 30000)
    assert target[1] == 20000
    assert target[0] * target[1] <= 25_000_000


def test_decode_guard_accepts_source_within_limit():
    assert validate_decode_pixels(1024, 30000, 50_000_000) == 30_720_000


@pytest.mark.parametrize(
    ("width", "height", "limit"),
    [(1024, 50000, 50_000_000), (0, 100, 50_000_000), (100, -1, 50_000_000)],
)
def test_decode_guard_rejects_unsafe_or_invalid_source(width, height, limit):
    with pytest.raises(ValueError):
        validate_decode_pixels(width, height, limit)


@pytest.mark.parametrize(
    ("width", "height", "max_height", "max_pixels"),
    [(0, 1, 100, 100), (1, 0, 100, 100), (1, 1, 0, 100), (1, 1, 100, 0)],
)
def test_fit_dimensions_rejects_invalid_limits(width, height, max_height, max_pixels):
    with pytest.raises(ValueError):
        fit_dimensions(width, height, max_height, max_pixels)
