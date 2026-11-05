import pytest

from core.pagination import collect_all_pages


def test_collect_all_pages_reads_every_page_in_order():
    pages = {
        None: {
            "files": [{"id": "a"}, {"id": "b"}],
            "nextPageToken": "page-2",
        },
        "page-2": {
            "files": [{"id": "c"}],
            "nextPageToken": "page-3",
        },
        "page-3": {"files": [{"id": "d"}]},
    }
    requested_tokens = []

    def fetch_page(token):
        requested_tokens.append(token)
        return pages[token]

    result = collect_all_pages(fetch_page)

    assert [item["id"] for item in result] == ["a", "b", "c", "d"]
    assert requested_tokens == [None, "page-2", "page-3"]


def test_collect_all_pages_handles_empty_folder():
    assert collect_all_pages(lambda token: {"files": []}) == []


def test_collect_all_pages_handles_missing_files_key():
    assert collect_all_pages(lambda token: {}) == []


def test_collect_all_pages_rejects_repeated_page_token():
    def fetch_page(token):
        return {"files": [], "nextPageToken": "repeat"}

    with pytest.raises(RuntimeError, match="Pagination token repeated"):
        collect_all_pages(fetch_page)
