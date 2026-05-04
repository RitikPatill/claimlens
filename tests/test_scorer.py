from __future__ import annotations

import pytest

from claimlens.models import ChunkVerdict, ClaimVerdict, Label
from claimlens.scorer import score


def _make_verdict(label: Label) -> ChunkVerdict:
    return ChunkVerdict(
        chunk_text="x",
        source_url="http://example.com",
        label=label,
        reasoning="r",
    )


def test_all_supports():
    verdicts = [_make_verdict(Label.SUPPORTS)] * 3
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.SUPPORTED
    assert confidence == 1.0


def test_all_refutes():
    verdicts = [_make_verdict(Label.REFUTES)] * 3
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.REFUTED
    assert confidence == 0.0


def test_all_neutral():
    verdicts = [_make_verdict(Label.NEUTRAL)] * 3
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert confidence == 0.5


def test_empty():
    verdict, confidence = score([])
    assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert confidence == 0.0


def test_mixed_supports_win():
    # 3 SUPPORTS, 1 REFUTES, 1 NEUTRAL → raw = (3-1)/5 = 0.4 → confidence = 0.7
    verdicts = (
        [_make_verdict(Label.SUPPORTS)] * 3
        + [_make_verdict(Label.REFUTES)]
        + [_make_verdict(Label.NEUTRAL)]
    )
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.SUPPORTED
    assert confidence == pytest.approx(0.7)


def test_tie():
    # 2 SUPPORTS, 2 REFUTES → raw = 0 → confidence = 0.5
    verdicts = (
        [_make_verdict(Label.SUPPORTS)] * 2
        + [_make_verdict(Label.REFUTES)] * 2
    )
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.INSUFFICIENT_EVIDENCE
    assert confidence == 0.5


def test_single_supports_chunk():
    # raw = (1-0)/1 = 1.0 → confidence = (1+1)/2 = 1.0
    verdict, confidence = score([_make_verdict(Label.SUPPORTS)])
    assert verdict == ClaimVerdict.SUPPORTED
    assert confidence == 1.0


def test_single_refutes_chunk():
    # raw = (0-1)/1 = -1.0 → confidence = (-1+1)/2 = 0.0
    verdict, confidence = score([_make_verdict(Label.REFUTES)])
    assert verdict == ClaimVerdict.REFUTED
    assert confidence == 0.0


def test_refutes_win():
    # 1 SUPPORTS, 3 REFUTES → raw = (1-3)/4 = -0.5 → confidence = (-0.5+1)/2 = 0.25
    verdicts = (
        [_make_verdict(Label.SUPPORTS)]
        + [_make_verdict(Label.REFUTES)] * 3
    )
    verdict, confidence = score(verdicts)
    assert verdict == ClaimVerdict.REFUTED
    assert confidence == pytest.approx(0.25)
