"""Generic pagination helpers used by external API adapters."""


def collect_all_pages(fetch_page):
    """Collect every page from a callback accepting a page token."""
    items = []
    page_token = None
    seen_tokens = set()

    while True:
        response = fetch_page(page_token) or {}
        items.extend(response.get("files", []))
        next_token = response.get("nextPageToken")
        if not next_token:
            return items
        if next_token in seen_tokens:
            raise RuntimeError(
                f"Pagination token repeated; refusing infinite loop: {next_token}"
            )
        seen_tokens.add(next_token)
        page_token = next_token
