![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-WIP-yellow)

# ClaimLens

ClaimLens is a locally-runnable claim verification pipeline that retrieves live web evidence for any plain-English claim and runs a structured LLM pass to label each source chunk as `SUPPORTS`, `REFUTES`, or `NEUTRAL`. It returns a fully structured JSON verdict with source attribution — no hallucination, no black box.

## Architecture

```
Claim ──► DuckDuckGo Search ──► Chunker + Embedder
                                       │
                                       ▼
                                 ChromaDB (ephemeral)
                                       │
                                       ▼
                            LLM Verifier (GPT-4o-mini / Claude Haiku)
                                       │
                                       ▼
                               Scorer / Aggregator
                                       │
                                       ▼
                        ClaimResult JSON (verdict + evidence)
                                       │
                                       ▼
                          FastAPI  /verify  ·  CLI (rich)
```

## What Works — M5 (current)

| Component | Status |
|---|---|
| `claimlens/retriever.py` | DuckDuckGo search — returns top-5 `{title, url, body}` dicts |
| `claimlens/chunker.py` | Overlapping 200-word sliding-window chunker; propagates source URL |
| `claimlens/embedder.py` | Ephemeral ChromaDB + `all-MiniLM-L6-v2` embeddings; `index_chunks()` + `query()` |
| `claimlens/models.py` | Pydantic schema — `Label`, `ClaimVerdict`, `ChunkVerdict`, `ClaimResult` |
| `claimlens/verifier.py` | OpenAI GPT-4o-mini structured-output pass; labels each chunk SUPPORTS/REFUTES/NEUTRAL |
| `claimlens/scorer.py` | Pure aggregation — confidence formula `(supports−refutes)/total` → [0,1]; resolves final verdict |
| `claimlens/pipeline.py` | `pipeline.run(claim)` wires all stages end-to-end → `ClaimResult` |
| `claimlens/api.py` | FastAPI `POST /verify` endpoint + `GET /health` |
| `claimlens/__main__.py` | CLI — pretty-prints verdict, confidence, top-3 evidence chunks (rich) |
| `tests/test_retrieval.py` | 8 unit tests covering chunker, retriever, and embedder roundtrip |
| `tests/test_scorer.py` | 6 pure unit tests for all scorer verdict branches and confidence formula |
| `tests/test_api.py` | 5 unit tests for `/verify` endpoint (mocked pipeline, happy path + 502) |

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

### API server

```bash
uvicorn claimlens.api:app --reload
# POST http://localhost:8000/verify
# Body: {"claim": "Germany had the highest GDP in the EU in 2023"}
```

### CLI

```bash
python -m claimlens "Germany had the highest GDP in the EU in 2023"
```

### Python API

```python
from claimlens.pipeline import run

result = run("Germany had the highest GDP in the EU in 2023")
print(result.verdict)       # SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE
print(result.confidence)    # 0.0 – 1.0
for ev in result.evidence:
    print(ev.label, ev.source_url)
    print(ev.chunk_text[:120])
```

> **First run:** `index_chunks()` auto-downloads `all-MiniLM-L6-v2` (~90 MB). Subsequent calls reuse the cached singleton.

### Retrieval pipeline smoke test

```python
from claimlens.retriever import search
from claimlens.chunker import chunk_results
from claimlens.embedder import index_chunks, query

results = search("Germany highest GDP EU 2023")
chunks = chunk_results(results)
col = index_chunks(chunks)
hits = query(col, "Germany highest GDP EU 2023")
print(hits[0])
# {"text": "...", "url": "https://...", "title": "..."}
```

## Project Layout

```
claimlens/
├── __init__.py        # package version  (M1)
├── __main__.py        # CLI — rich verdict + evidence table  (M5)
├── retriever.py       # DuckDuckGo search  (M2)
├── chunker.py         # overlapping sliding-window chunker  (M2)
├── embedder.py        # sentence-transformers + ChromaDB  (M2)
├── verifier.py        # LLM verification pass  (M3)
├── models.py          # Pydantic output schema  (M3)
├── scorer.py          # label aggregation + confidence  (M4)
├── pipeline.py        # end-to-end run(claim) orchestrator  (M4)
└── api.py             # FastAPI /verify + /health endpoints  (M5)
tests/
├── __init__.py        (M1)
├── test_retrieval.py  # chunker, retriever, embedder roundtrip — 8 tests  (M2)
├── test_scorer.py     # scorer verdict branches and confidence formula — 6 tests  (M4)
├── test_api.py        # /verify endpoint — 5 tests with mocked pipeline  (M5)
└── test_verifier.py   # mock-based LLM verifier tests  (M6 — planned)
```

## Environment Variables

Set one of the following before running M3+ features:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for GPT-4o-mini backend (default) |
| `ANTHROPIC_API_KEY` | Required when `LLM_BACKEND=anthropic` |
| `LLM_BACKEND` | Set to `anthropic` to use Claude Haiku instead of GPT-4o-mini |

## Running Tests

```bash
pytest tests/
```

14 tests pass with no network or model calls (all external I/O is mocked).

## Roadmap

| Milestone | Scope |
|---|---|
| M1 | Project scaffold, CLI stub, pinned dependencies |
| M2 | Retrieval pipeline — search, chunker, embedder, ChromaDB |
| M3 | LLM verification pass — structured output, Pydantic models |
| M4 | Attribution scoring — confidence formula, verdict aggregation |
| M5 | FastAPI endpoint, argparse CLI with rich output ← **current** |
| M6 | Verifier unit tests, edge-case scorer coverage |

## License

MIT — see [LICENSE](LICENSE).
