from PIL import Image

from core.local_paths import default_output_dir, discover_images
from local_runner import _save_image


def test_default_output_is_inside_input_named_output_cleaned(tmp_path):
    source = tmp_path / "Chapter 01"
    assert default_output_dir(source) == source / "output_cleaned"


def test_discover_images_filters_and_naturally_sorts(tmp_path):
    (tmp_path / "page10.png").write_bytes(b"x")
    (tmp_path / "page2.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore")
    (tmp_path / "page1.WEBP").write_bytes(b"x")
    (tmp_path / "output_cleaned").mkdir()
    (tmp_path / "output_cleaned" / "old.png").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    assert [p.name for p in discover_images(tmp_path)] == [
        "page1.WEBP", "page2.jpg", "page10.png"
    ]


def test_discover_images_does_not_recurse(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "page1.png").write_bytes(b"x")
    assert discover_images(tmp_path) == []


def test_save_png_preserves_alpha_channel(tmp_path):
    source = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    source.putpixel((1, 1), (0, 255, 0, 128))
    destination = tmp_path / "page.png"

    _save_image(source, destination)

    with Image.open(destination) as saved:
        assert saved.mode == "RGBA"
        assert saved.getpixel((0, 0))[3] == 0
        assert saved.getpixel((1, 1))[3] == 128
