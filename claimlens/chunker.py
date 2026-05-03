from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split *text* into overlapping word-based chunks.

    Uses whitespace tokenisation (no external dependencies).
    Step size is ``chunk_size - overlap`` words.
    Returns an empty list for empty input.
    Inputs shorter than *chunk_size* words produce exactly one chunk.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        # Stop once this chunk already reached the end to avoid a tiny tail chunk.
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_results(
    results: list[dict],
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[dict]:
    """Chunk each search result's body text, preserving source metadata.

    For each result dict ``{"title", "url", "body"}``, produces N chunk dicts:
    ``{"text": str, "url": str, "title": str}``.
    Results with empty bodies are skipped.
    """
    chunks: list[dict] = []
    for result in results:
        body = result.get("body") or ""
        if not body:
            continue
        for text in chunk_text(body, chunk_size=chunk_size, overlap=overlap):
            chunks.append(
                {
                    "text": text,
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                }
            )
    return chunks
