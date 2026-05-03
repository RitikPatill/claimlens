from __future__ import annotations

from claimlens.models import ChunkVerdict, ClaimVerdict, Label


def score(verdicts: list[ChunkVerdict]) -> tuple[ClaimVerdict, float]:
    """Aggregate per-chunk labels into a final verdict and confidence score.

    Confidence is computed as ``(supports - refutes) / total`` mapped to
    [0, 1] via ``(raw + 1) / 2``.

    Args:
        verdicts: List of :class:`ChunkVerdict` objects from the verifier.

    Returns:
        A tuple of ``(ClaimVerdict, confidence)`` where confidence is
        rounded to 4 decimal places and clamped to [0.0, 1.0].
    """
    total = len(verdicts)
    if total == 0:
        return ClaimVerdict.INSUFFICIENT_EVIDENCE, 0.0

    supports = sum(1 for v in verdicts if v.label == Label.SUPPORTS)
    refutes = sum(1 for v in verdicts if v.label == Label.REFUTES)

    raw = (supports - refutes) / total          # ∈ [-1.0, 1.0]
    confidence = round((raw + 1.0) / 2.0, 4)   # ∈ [0.0, 1.0]

    if supports > refutes:
        verdict = ClaimVerdict.SUPPORTED
    elif refutes > supports:
        verdict = ClaimVerdict.REFUTED
    else:
        verdict = ClaimVerdict.INSUFFICIENT_EVIDENCE

    return verdict, confidence
