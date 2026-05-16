# Bhagavad-Gita — Engineering README

This repository implements a production-oriented retrieval assistant over the Bhagavad-Gita corpus. The README is written for engineers and reviewers: it explains the architecture, design decisions, reproducibility steps, observability, and extensibility points that make this project review-ready.

**What reviewers should notice**
- **Reproducibility:** offline embedding pipeline with deterministic ingestion.
- **Separation of concerns:** ingestion, embedding, retrieval, and runtime are decoupled for independent scaling.
- **Observability & evaluation:** evaluation harness and outputs are included for measurable quality checks.
- **Extensibility:** clear abstraction points for embedding providers, vector stores, and runtime responders.

**Repository layout (quick)**

- [data/](data/) — source JSON: `verses.json`, `purports.json`, `metadata.json`.
- [frontend/](frontend/) — minimal UI and example integrations (`gita_assistant.html`, `index.html`).
- [python-backend/](python-backend/) — ingestion, embedding interface, and search utilities.
- [runtime/](runtime/) — Node.js orchestration: pipeline, intent classification, and runtime adapters.
- [tools/](tools/) — evaluation harnesses and test queries.
- [outputs/](outputs/) — generated artifacts: `answers.json`, `history.json`, `retrieval_eval_results.json`.

**Architecture (concise)**

```mermaid
flowchart LR
	A[Raw JSON data] --> B[Ingestion]
	B --> C[Document store + metadata]
	C --> D[Embedding pipeline]
	D --> E[Vector index]
	E --> F[Runtime retrieval (kNN)]
	F --> G[Response composition + UI]
```

Key runtime pieces:
- `python-backend/loader.py` — deterministic ingestion and document normalization.
- `python-backend/embeddings.py` — embedding abstraction; currently wired to offline provider but designed for pluggable backends.
- `python-backend/search.py` — retrieval primitives and filtering.
- `runtime/pipeline.js` — orchestrates retrieval, intent classification, and response construction.

**Design principles**
- Single-responsibility modules: keep ingestion, embedding, and serving orthogonal.
- Idempotent ingest: re-running the pipeline must produce the same canonical document IDs and vectors.
- Testable evaluation: retrieval tests and metrics live in `tools/eval/` and output to `outputs/` for CI checks.
- Config-first: model and index parameters live in small config objects (easy to surface to env/CI).

**Developer quickstart**

Prerequisites: Python 3.8+, Node 16+, `npm` or `yarn`.

1) Install Python deps:

```bash
cd python-backend
python -m pip install -r requirements.txt
```

2) (Optional) Install Node deps for runtime tooling:

```bash
npm install
```

3) Run ingestion + embeddings (reproducible):

```bash
cd python-backend
python loader.py         # normalize and export documents
python embeddings.py     # generate vector embeddings (offline)
```

4) Start the local server for UI/testing:

```bash
node server.js
# then open http://localhost:3000
```

**Evaluation & metrics**

- Use `tools/eval/retrieval_eval.js` with `tools/eval/retrieval_test_queries.json` to compute retrieval accuracy and expose regression when changing models or tokenizers.
- Results are recorded to `outputs/retrieval_eval_results.json` to support PR-level gates.

**Observability & QA**
- Emit deterministic artifacts: ingested documents, vector dumps, and top-k retrievals.
- Add lightweight logging in `runtime/pipeline.js` and `python-backend/search.py` with structured JSON for easy ingestion into log pipelines.
- Use `outputs/history.json` and `outputs/answers.json` for replay-based QA.

**Performance & scaling notes**
- Keep embedding generation offline and shardable (parallelize by file/chunk in `loader.py`).
- Vector store abstraction allows swapping between an in-memory ANN (faiss/hnswlib) and a managed service.
- Cache hot queries at the runtime layer to reduce retrieval QPS.

**Security & privacy**
- No secrets are checked into source. Any provider keys should be injected via env vars in CI/containers.
- Validate and sanitize user inputs at the runtime boundary to avoid injection into templates.

**Files reviewers should inspect**
- [python-backend/loader.py](python-backend/loader.py) — ingestion logic and canonicalization.
- [python-backend/embeddings.py](python-backend/embeddings.py) — embedding interface and adapters.
- [python-backend/search.py](python-backend/search.py) — retrieval and ranking heuristics.
- [runtime/pipeline.js](runtime/pipeline.js) — orchestration and response assembly.
- [tools/eval/retrieval_eval.js](tools/eval/retrieval_eval.js) — evaluation harness and metrics.

**Recommended next engineering steps (high-impact)**
1. Add CI that runs `python-backend/loader.py` + `embeddings.py` and checks that `outputs/retrieval_eval_results.json` does not regress.
2. Add a `Dockerfile` for `python-backend` and `runtime` and a `docker-compose.yml` for local integration testing.
3. Add `pytest` tests for ingestion normalization and a small retrieval integration test.

---