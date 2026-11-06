from core.local_paths import default_output_dir, discover_images


def test_default_output_is_sibling_named_cleaned(tmp_path):
    source = tmp_path / "Chapter 01"
    assert default_output_dir(source) == tmp_path / "Chapter 01_cleaned"


def test_discover_images_filters_and_naturally_sorts(tmp_path):
    (tmp_path / "page10.png").write_bytes(b"x")
    (tmp_path / "page2.jpg").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore")
    (tmp_path / "page1.WEBP").write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    assert [p.name for p in discover_images(tmp_path)] == [
        "page1.WEBP", "page2.jpg", "page10.png"
    ]


def test_discover_images_does_not_recurse(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "page1.png").write_bytes(b"x")
    assert discover_images(tmp_path) == []
