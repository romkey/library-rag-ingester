# library-rag-ingester overview

**What:** mirrors the PDX Hackerspace library catalog (one Qdrant point per book) into a Qdrant collection for RAG.
**Runs as:** worker that syncs every `POLL_INTERVAL_SECONDS` (or once, `--once`). No port; liveness is a heartbeat file.
**Talks to:** the library app's `GET /api/v1/catalog` (JSON Lines, Bearer API key), an OpenAI-compatible embeddings API, Qdrant.

## Files

- `src/library_rag_ingester/__main__.py`: command line (sync, `query`, `healthcheck`), logging, SIGTERM, wiring.
- `config.py`: `Settings` from environment variables.
- `catalog.py`: `CatalogClient` (Bearer key, ETag/304) and the JSONL parser into `Book`.
- `documents.py`: pure functions: `Book` → `Document` (embedding text, payload, content hash, point id).
- `embeddings.py`: `OpenAIEmbedder`, with query/document prompt prefixes and truncation.
- `qdrant_store.py`: collection setup, payload indexes, stored hashes, upsert, delete, search.
- `service.py`: the sync cycle and polling loop.
- `heartbeat.py`: heartbeat file for the production health check.
- `query.py`: search and result formatting for `query`.
- `tests/`: pytest; `fakes.py` has a fake catalog and embedder, `conftest.py` an in-memory Qdrant.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Qdrant), `.test`, `.lint`, `.prod` (with Qdrant).

## Sync invariants

- Point id = uuid5 of `library:book:<id>`; never change the scheme without a migration plan, or points duplicate.
- Each point's `content_hash` covers the text, payload, model, document prefix and `DOCUMENT_SCHEMA_VERSION`; only books whose hash changed are embedded.
- Every sync fetches the whole catalog (the feed doesn't report deletions), so stored books missing from it are deleted.
- A stopped (SIGTERM) cycle, or an empty catalog, never deletes anything.
- A 304 (unchanged ETag) skips the cycle. The ETag is kept only after a cycle finishes.
- Bump `DOCUMENT_SCHEMA_VERSION` in `documents.py` when document text or payload layout changes.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
