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
```

## What Works — M1

| Component | Status |
|---|---|
| `claimlens/__init__.py` | Package created; `__version__ = "0.1.0"` |
| `claimlens/__main__.py` | CLI stub — accepts a claim string, reports pipeline not yet implemented |
| `requirements.txt` | All production and test dependencies pinned |
| `tests/__init__.py` | Test package initialized |
| `LICENSE`, `.gitignore` | Present |

The package installs cleanly and `python -m claimlens "any claim"` runs without error. No retrieval, embedding, or verification logic is wired yet.

## Quick Start

```bash
pip install -r requirements.txt
```

**CLI** (coming in M5):
```bash
python -m claimlens "Germany had the highest GDP in the EU in 2023"
```

**API** (coming in M5):
```bash
uvicorn claimlens.api:app --reload
# POST http://localhost:8000/verify
# Body: {"claim": "Germany had the highest GDP in the EU in 2023"}
```

## Project Layout

```
claimlens/
├── __init__.py        # package version  (M1)
├── __main__.py        # CLI entry-point stub  (M1)
├── retriever.py       # DuckDuckGo search + chunking  (M2)
├── embedder.py        # sentence-transformers + ChromaDB  (M2)
├── verifier.py        # LLM verification pass  (M3)
├── scorer.py          # label aggregation + confidence  (M4)
├── models.py          # Pydantic output models  (M3)
└── api.py             # FastAPI /verify endpoint  (M5)
tests/
├── __init__.py        (M1)
├── test_scorer.py     (M4)
└── test_verifier.py   (M3)
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

## License

MIT — see [LICENSE](LICENSE).
