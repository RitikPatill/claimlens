from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Module-level lazy singleton — avoids re-downloading/re-loading the ~90 MB
# model on every call while keeping module import fast.
_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    """Return the cached SentenceTransformer, loading it on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def index_chunks(
    chunks: list[dict],
    collection_name: str = "claimlens",
) -> chromadb.Collection:
    """Embed *chunks* and store them in an ephemeral ChromaDB collection.

    Each chunk dict must have keys ``"text"``, ``"url"``, and ``"title"``.
    A uuid4 suffix is appended to *collection_name* so concurrent calls
    within the same process don't collide.

    Returns the populated :class:`chromadb.Collection`.
    Empty *chunks* returns an empty collection without error.
    """
    import uuid as _uuid

    name = f"{collection_name}-{_uuid.uuid4().hex}"
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=name, embedding_function=None)

    if not chunks:
        return collection

    model = _get_model()
    texts = [c["text"] for c in chunks]
    metadatas = [{"url": c.get("url", ""), "title": c.get("title", "")} for c in chunks]
    ids = [str(i) for i in range(len(chunks))]

    embeddings = model.encode(texts).tolist()  # numpy array → plain Python list

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return collection


def query(
    collection: chromadb.Collection,
    claim: str,
    n_results: int = 5,
) -> list[dict]:
    """Semantic search over *collection*.

    Embeds *claim* via the shared model and retrieves the *n_results* nearest
    chunks. *n_results* is clamped to ``collection.count()`` to avoid a
    ChromaDB error when the collection is smaller than requested.

    Returns a list of dicts:
    ``{"text": str, "url": str, "title": str, "distance": float}``.
    Returns ``[]`` if the collection is empty.
    """
    count = collection.count()
    if count == 0:
        return []

    k = min(n_results, count)

    model = _get_model()
    query_embedding = model.encode([claim]).tolist()

    result = collection.query(query_embeddings=query_embedding, n_results=k)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        {
            "text": doc,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
