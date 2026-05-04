from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from claimlens.api import app
from claimlens.models import ChunkVerdict, ClaimResult, ClaimVerdict, Label

client = TestClient(app)


def _make_result() -> ClaimResult:
    return ClaimResult(
        claim="test claim",
        verdict=ClaimVerdict.SUPPORTED,
        confidence=0.75,
        evidence=[
            ChunkVerdict(
                chunk_text="Some supporting text about the claim.",
                source_url="https://example.com/article",
                label=Label.SUPPORTS,
                reasoning="The text directly supports the claim.",
            )
        ],
    )


def test_verify_returns_200(mocker):
    mocker.patch("claimlens.api.run", return_value=_make_result())
    response = client.post("/verify", json={"claim": "test claim"})
    assert response.status_code == 200
    body = response.json()
    assert "claim" in body
    assert "verdict" in body
    assert "confidence" in body
    assert "evidence" in body


def test_verify_verdict_field(mocker):
    result = _make_result()
    mocker.patch("claimlens.api.run", return_value=result)
    response = client.post("/verify", json={"claim": "test claim"})
    assert response.json()["verdict"] == result.verdict.value


def test_verify_missing_claim_returns_422(mocker):
    mock_run = mocker.patch("claimlens.api.run")
    response = client.post("/verify", json={})
    assert response.status_code == 422
    mock_run.assert_not_called()


def test_verify_empty_claim_calls_pipeline(mocker):
    mock_run = mocker.patch("claimlens.api.run", return_value=_make_result())
    response = client.post("/verify", json={"claim": ""})
    assert response.status_code == 200
    mock_run.assert_called_once_with("")


def test_verify_pipeline_error_returns_502(mocker):
    mocker.patch("claimlens.api.run", side_effect=RuntimeError("search failed"))
    response = client.post("/verify", json={"claim": "test claim"})
    assert response.status_code == 502
    assert "search failed" in response.json()["detail"]
