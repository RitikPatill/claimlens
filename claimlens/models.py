from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Label(str, Enum):
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEUTRAL = "NEUTRAL"


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ChunkVerdict(BaseModel):
    chunk_text: str
    source_url: str
    label: Label
    reasoning: str


class ClaimResult(BaseModel):
    claim: str
    verdict: ClaimVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[ChunkVerdict]
