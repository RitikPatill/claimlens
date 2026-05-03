from __future__ import annotations

import openai
from pydantic import BaseModel

from claimlens.models import ChunkVerdict, Label


# Private models used only for the structured LLM response
class _ChunkLabel(BaseModel):
    label: Label
    reasoning: str


class _LabelBatch(BaseModel):
    labels: list[_ChunkLabel]


def verify(
    claim: str,
    chunks: list[dict],
    model: str = "gpt-4o-mini",
) -> list[ChunkVerdict]:
    """Label each chunk in *chunks* relative to *claim*.

    Calls the OpenAI structured-output API and returns one :class:`ChunkVerdict`
    per chunk (or fewer if the model returns fewer labels than chunks).

    Args:
        claim: The plain-English claim to verify.
        chunks: List of dicts with keys ``"text"``, ``"url"``, ``"title"``
                as returned by :func:`claimlens.embedder.query`.
        model: OpenAI model name. Defaults to ``"gpt-4o-mini"``.

    Returns:
        List of :class:`ChunkVerdict` objects, one per chunk (zipped with
        the LLM response so length mismatches are handled safely).
    """
    if not chunks:
        return []

    numbered = "\n\n".join(
        f"[{i}] {chunk['text']}" for i, chunk in enumerate(chunks)
    )

    system_prompt = (
        "You are a fact-checking assistant. "
        "For each numbered text chunk, decide whether it SUPPORTS, REFUTES, "
        "or is NEUTRAL toward the given claim. "
        "Return exactly one label per chunk in the same order."
    )
    user_prompt = (
        f"Claim: {claim}\n\n"
        f"Chunks:\n{numbered}\n\n"
        f"Return a JSON object with a 'labels' array containing one entry per chunk, "
        f"each with 'label' (SUPPORTS/REFUTES/NEUTRAL) and 'reasoning' (one sentence)."
    )

    client = openai.OpenAI()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=_LabelBatch,
    )

    batch: _LabelBatch = completion.choices[0].message.parsed

    return [
        ChunkVerdict(
            chunk_text=chunk["text"],
            source_url=chunk.get("url", ""),
            label=item.label,
            reasoning=item.reasoning,
        )
        for item, chunk in zip(batch.labels, chunks)
    ]
