from PIL import Image
from core.runpod_client import _build_payload, normalize_b64


def test_build_payload_contains_jpeg_data_uri_and_filename():
    payload = _build_payload(Image.new("RGB", (4, 4), "white"), filename="page1.png")
    image_data = payload["input"]["image"]
    assert image_data.startswith("data:image/jpeg;base64,")
    assert payload["input"]["filename"] == "page1.png"


def test_normalize_base64_removes_data_uri_and_padding():
    assert normalize_b64("data:image/png;base64,YWJj") == "YWJj"
    assert normalize_b64("YWI") == "YWI="
