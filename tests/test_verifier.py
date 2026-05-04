from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from claimlens.models import Label
from claimlens.verifier import verify


def _mock_openai(mocker, labels: list[tuple[Label, str]]) -> MagicMock:
    items = [MagicMock(label=lbl, reasoning=rsn) for lbl, rsn in labels]
    batch = MagicMock(labels=items)
    completion = MagicMock()
    completion.choices[0].message.parsed = batch
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = completion
    mocker.patch("claimlens.verifier.openai.OpenAI", return_value=mock_client)
    return mock_client


def _chunk(text: str = "some text", url: str = "http://example.com") -> dict:
    return {"text": text, "url": url, "title": "title"}


def test_empty_chunks_returns_empty_list(mocker):
    mock_openai_class = mocker.patch("claimlens.verifier.openai.OpenAI")
    result = verify("some claim", [])
    assert result == []
    mock_openai_class.assert_not_called()


def test_single_chunk_supports(mocker):
    _mock_openai(mocker, [(Label.SUPPORTS, "It clearly supports.")])
    result = verify("some claim", [_chunk()])
    assert len(result) == 1
    assert result[0].label == Label.SUPPORTS


def test_multiple_chunks_mixed_labels(mocker):
    labels = [
        (Label.SUPPORTS, "Supports the claim."),
        (Label.REFUTES, "Refutes the claim."),
        (Label.NEUTRAL, "Neither supports nor refutes."),
    ]
    _mock_openai(mocker, labels)
    chunks = [_chunk(f"text {i}") for i in range(3)]
    result = verify("some claim", chunks)
    assert len(result) == 3
    assert result[0].label == Label.SUPPORTS
    assert result[1].label == Label.REFUTES
    assert result[2].label == Label.NEUTRAL


def test_source_url_and_text_propagated(mocker):
    _mock_openai(mocker, [(Label.NEUTRAL, "Neutral.")])
    chunk = _chunk(text="unique text content", url="https://source.example.org/page")
    result = verify("some claim", [chunk])
    assert result[0].chunk_text == "unique text content"
    assert result[0].source_url == "https://source.example.org/page"


def test_model_name_forwarded_to_api(mocker):
    mock_client = _mock_openai(mocker, [(Label.SUPPORTS, "Supports.")])
    verify("some claim", [_chunk()], model="gpt-4-turbo")
    call_kwargs = mock_client.beta.chat.completions.parse.call_args
    assert call_kwargs.kwargs["model"] == "gpt-4-turbo"


def test_label_count_mismatch_truncates_safely(mocker):
    # LLM returns 2 labels for 3 chunks — zip truncates to 2
    _mock_openai(mocker, [(Label.SUPPORTS, "S."), (Label.REFUTES, "R.")])
    chunks = [_chunk(f"text {i}") for i in range(3)]
    result = verify("some claim", chunks)
    assert len(result) == 2
    assert result[0].label == Label.SUPPORTS
    assert result[1].label == Label.REFUTES


def test_missing_url_fallback(mocker):
    # chunk dict has no "url" key — source_url should default to ""
    _mock_openai(mocker, [(Label.NEUTRAL, "No URL.")])
    chunk = {"text": "Some text without a URL.", "title": "No URL"}
    result = verify("some claim", [chunk])
    assert len(result) == 1
    assert result[0].source_url == ""
