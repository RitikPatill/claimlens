# ClaimLens


> **Video walkthrough:** https://youtu.be/F7in_X7VxH0
> **60-second overview:** https://youtu.be/jS9ruO9IB7g

> Verify any claim against live web evidence — local RAG pipeline with per-source attribution scores and verdict confidence.

<!-- TODO: replace with a 5-10 second demo gif. Record with ScreenToGif on
     Windows or peek on macOS. Save to docs/demo.gif and update path here. -->
![demo](docs/demo.gif)

## What it is

ClaimLens is a locally-runnable claim verification pipeline. Given a plain-English claim, it fetches the top-5 web results via DuckDuckGo (no API key required), chunks and embeds the text into an ephemeral ChromaDB vector store using sentence-transformers, then runs a structured LLM pass (GPT-4o-mini) that labels each chunk as `SUPPORTS`, `REFUTES`, or `NEUTRAL` relative to the claim. Those per-chunk labels are aggregated into a final verdict — `SUPPORTED`, `REFUTED`, or `INSUFFICIENT_EVIDENCE` — alongside a confidence score between 0 and 1.

Every response includes a structured evidence list: source URL, quoted chunk text, label, and the model's reasoning for that label. The LLM only classifies text retrieved from the web; it does not generate factual claims of its own.

## Quickstart

```bash
git clone https://github.com/RitikPatill/claimlens.git
cd claimlens
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python -m claimlens "Germany had the highest GDP in the EU in 2023"
```

> The first run downloads the `all-MiniLM-L6-v2` sentence-transformers model (~90 MB). Subsequent calls reuse the cached copy.

## Usage

**CLI** — runs the full pipeline and prints a verdict panel with a top-3 evidence table:

```bash
python -m claimlens "SpaceX was the first private company to dock with the ISS"
```

**API server** — start the FastAPI server, then POST a claim:

```bash
uvicorn claimlens.api:app --reload
```

```bash
curl -s -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "Germany had the highest GDP in the EU in 2023"}' | python -m json.tool
```

The response is a JSON object with `claim`, `verdict`, `confidence`, and an `evidence` array. Each evidence entry contains `source_url`, `chunk_text`, `label`, and `reasoning`.

**Python** — call the pipeline directly:

```python
from claimlens.pipeline import run

result = run("Germany had the highest GDP in the EU in 2023")
print(result.verdict)      # SUPPORTED / REFUTED / INSUFFICIENT_EVIDENCE
print(result.confidence)   # 0.0 – 1.0
for ev in result.evidence:
    print(ev.label, ev.source_url)
```

## Architecture

```
Claim ──► DuckDuckGo Search ──► Chunker + Embedder (all-MiniLM-L6-v2)
                                          │
                                          ▼
                                  ChromaDB (ephemeral)
                                          │
                                          ▼
                               LLM Verifier (GPT-4o-mini)
                               SUPPORTS / REFUTES / NEUTRAL per chunk
                                          │
                                          ▼
                               Scorer → verdict + confidence
                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                        FastAPI /verify          CLI (rich)
```

## Project structure

```
claimlens/
├── __main__.py        CLI — argparse entry point, rich verdict panel and evidence table
├── api.py             FastAPI app — POST /verify, GET /health
├── pipeline.py        run(claim) — wires all stages end-to-end
├── retriever.py       DuckDuckGo search, returns top-5 {title, url, body} dicts
├── chunker.py         Overlapping 200-word sliding-window chunker, propagates source URL
├── embedder.py        sentence-transformers + ephemeral ChromaDB indexing and query
├── verifier.py        OpenAI GPT-4o-mini structured-output LLM verification pass
├── scorer.py          Label aggregation — confidence formula and final verdict resolution
└── models.py          Pydantic schema — Label, ChunkVerdict, ClaimVerdict, ClaimResult
tests/
├── test_retrieval.py  8 unit tests — chunker, retriever, embedder roundtrip (all mocked)
├── test_verifier.py   7 unit tests — LLM call, label propagation, edge cases (mocked)
├── test_scorer.py     9 pure unit tests — verdict branches, confidence formula, edge cases
└── test_api.py        5 unit tests — /verify and /health endpoints with mocked pipeline
```

Run the full suite (29 tests, no network or model calls):

```bash
pytest tests/
```

## Roadmap

- [ ] Async batch processing — accept a list of claims in a single API request
- [ ] Persistent vector store — swap ephemeral ChromaDB for PostgreSQL + pgvector to cache embeddings across sessions
- [ ] Anthropic Claude Haiku backend — wire native tool-use API as a drop-in alternative to OpenAI
- [ ] Confidence calibration — benchmark verdict accuracy against a labelled fact-checking dataset (e.g. FEVER)
- [ ] Interactive web UI — claim input form with collapsible per-source evidence cards

## License

MIT — see [LICENSE](LICENSE).

---

Built autonomously by [autodev](https://github.com/RitikPatill/autodev),
a multi-agent orchestrator I designed. Each commit in this repo was
authored by me; the implementation work was performed by Sonnet under
the orchestrator's control. Read the orchestrator's README to see how.
