from __future__ import annotations

from claimlens import chunker, embedder, retriever, scorer, verifier
from claimlens.models import ClaimResult


def run(claim: str) -> ClaimResult:
    """Run the full claim-verification pipeline.

    Stages:
    1. Retrieve top web results for *claim* via DuckDuckGo.
    2. Chunk each result body into overlapping word windows.
    3. Embed chunks into an ephemeral ChromaDB collection.
    4. Query the collection for the most relevant chunks.
    5. Ask the LLM to label each chunk as SUPPORTS / REFUTES / NEUTRAL.
    6. Aggregate labels into a final verdict and confidence score.

    Args:
        claim: Plain-English claim to verify.

    Returns:
        A :class:`ClaimResult` with verdict, confidence, and per-chunk evidence.

    Raises:
        RuntimeError: If DuckDuckGo search fails.
        openai.OpenAIError: If the LLM call fails.
    """
    results = retriever.search(claim)
    chunks = chunker.chunk_results(results)
    collection = embedder.index_chunks(chunks)
    relevant = embedder.query(collection, claim)
    verdicts = verifier.verify(claim, relevant)
    verdict, confidence = scorer.score(verdicts)
    return ClaimResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        evidence=verdicts,
    )
