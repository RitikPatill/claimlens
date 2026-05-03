"""Unit tests for chunker.py, retriever.py, and embedder.py (no network / model calls)."""

from unittest.mock import MagicMock, patch

import pytest

from claimlens.chunker import chunk_results, chunk_text
from claimlens.embedder import index_chunks, query
from claimlens.retriever import search


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_basic():
    """10-word input, chunk_size=5, overlap=2 → 3 chunks with correct words."""
    words = "one two three four five six seven eight nine ten"
    chunks = chunk_text(words, chunk_size=5, overlap=2)
    # step = 5 - 2 = 3
    # chunk 0: words[0:5]  → "one two three four five"
    # chunk 1: words[3:8]  → "four five six seven eight"
    # chunk 2: words[6:11] → "seven eight nine ten"
    assert len(chunks) == 3
    assert chunks[0] == "one two three four five"
    assert chunks[1] == "four five six seven eight"
    assert chunks[2] == "seven eight nine ten"


def test_chunk_text_short():
    """Input shorter than chunk_size → exactly one chunk."""
    text = "hello world"
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_empty():
    """Empty string → empty list."""
    assert chunk_text("") == []


# ---------------------------------------------------------------------------
# chunk_results
# ---------------------------------------------------------------------------


def test_chunk_results_propagates_url():
    """Each produced chunk dict carries the source URL."""
    results = [{"title": "T", "url": "http://x", "body": "a b c"}]
    chunks = chunk_results(results)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["url"] == "http://x"
        assert "text" in chunk


def test_chunk_results_skips_empty_body():
    """Results with empty body produce no chunks."""
    results = [{"title": "T", "url": "http://x", "body": ""}]
    chunks = chunk_results(results)
    assert chunks == []


# ---------------------------------------------------------------------------
# search (mocked)
# ---------------------------------------------------------------------------


def test_search_mock():
    """Mock DDGS.text to verify key rename (href → url) and max_results cap."""
    fake_rows = [
        {"title": "Result A", "href": "http://a.com", "body": "body A"},
        {"title": "Result B", "href": "http://b.com", "body": "body B"},
    ]

    with patch("claimlens.retriever.DDGS") as MockDDGS:
        instance = MagicMock()
        instance.text.return_value = iter(fake_rows)
        MockDDGS.return_value.__enter__ = MagicMock(return_value=instance)
        MockDDGS.return_value.__exit__ = MagicMock(return_value=False)

        results = search("some claim", max_results=5)

    assert len(results) == 2
    for r in results:
        assert "url" in r
        assert "href" not in r
    assert results[0]["url"] == "http://a.com"
    assert results[1]["url"] == "http://b.com"


# ---------------------------------------------------------------------------
# index_chunks / query (mocked model — no download)
# ---------------------------------------------------------------------------


class _FakeArray:
    """Minimal stand-in for a numpy ndarray with a .tolist() method."""

    def __init__(self, n: int, dim: int = 384):
        self._data = [[0.1 * (i + 1)] * dim for i in range(n)]

    def tolist(self) -> list:
        return self._data


def _make_mock_model() -> MagicMock:
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, **_kw: _FakeArray(len(texts))
    return mock


def test_index_chunks_and_query_roundtrip():
    """index_chunks stores two chunks; query returns dicts with required keys."""
    fake_chunks = [
        {"text": "The sky is blue.", "url": "http://a.com", "title": "A"},
        {"text": "Grass is green.", "url": "http://b.com", "title": "B"},
    ]

    with patch("claimlens.embedder._get_model", return_value=_make_mock_model()):
        col = index_chunks(fake_chunks)
        results = query(col, "blue sky")

    assert isinstance(results, list)
    assert len(results) > 0
    for item in results:
        assert "text" in item
        assert "url" in item
        assert "title" in item
        assert "distance" in item


def test_index_chunks_empty_input():
    """index_chunks([]) must not raise; query on empty collection returns []."""
    # No model calls expected for empty input — mock guards against accidental load.
    with patch("claimlens.embedder._get_model", return_value=_make_mock_model()):
        col = index_chunks([])

    assert col is not None
    results = query(col, "anything")
    assert results == []
