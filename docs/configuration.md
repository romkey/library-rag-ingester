# Configuration

library-rag-ingester reads these environment variables at startup. Put them
in `.env` (copy `.env.example`); compose passes it to the container. An
invalid value stops the app with exit status 2 and a message naming the
variable. An empty value means the default.

## Library

| Variable | Default | Meaning |
|----------|---------|---------|
| `LIBRARY_CATALOG_URL` | required | The library's catalog feed, `https://<library>/api/v1/catalog` (`http://` or `https://`) |
| `LIBRARY_API_KEY` | required | API key for the feed, sent as `Authorization: Bearer <key>`. An admin creates one in the library app under **Settings → API keys**. Secret. |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Timeout for catalog requests (1 or more). Embedding requests get at least 120. |

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. The compose files set it to their own Qdrant. |
| `QDRANT_API_KEY` | none | Qdrant API key, if the server needs one. Secret. |
| `QDRANT_COLLECTION` | `library` | Collection to create and sync |
| `QDRANT_DISTANCE` | `cosine` | Vector distance for a new collection: `cosine`, `dot`, `euclid` or `manhattan` |

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `nomic-ai/modernbert-embed-base` | Model the API at `OPENAI_BASE_URL` serves. Changing it changes the vector size, so use a new `QDRANT_COLLECTION` (or drop the old one); the app refuses to start against a collection of the wrong size. |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding request (1 or more) |
| `EMBEDDING_CONTEXT_LENGTH` | `512` | The model's token limit; longer texts are truncated, prefix included (1 or more) |
| `EMBEDDING_QUERY_PREFIX` | `search_query:` | Prompt prefix the model expects on search queries (the `query` command). `none` for no prefix. |
| `EMBEDDING_DOCUMENT_PREFIX` | `search_document:` | Prompt prefix the model expects on documents; books are embedded with it. `none` for no prefix. |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API: a local server (text-embeddings-inference, Ollama at `http://host:11434/v1`, vLLM, LM Studio) or `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | none | Sent as a bearer token. Required for api.openai.com. Secret. |

### Prompt prefixes

Many embedding models were trained with a task prefix on queries, on the
documents being searched, or both, and retrieve noticeably worse without
it. Set the two prefixes to match `EMBEDDING_MODEL`:

| Model | `EMBEDDING_QUERY_PREFIX` | `EMBEDDING_DOCUMENT_PREFIX` |
|-------|--------------------------|-----------------------------|
| `nomic-ai/modernbert-embed-base` (default), `nomic-embed-text` | `search_query:` | `search_document:` |
| e5 family (`intfloat/e5-*`) | `query:` | `passage:` |
| `mxbai-embed-large` | `Represent this sentence for searching relevant passages:` | `none` |
| OpenAI `text-embedding-3-*` | `none` | `none` |

A space is added after a prefix unless it already ends in whitespace, and
`\n` becomes a newline. The model and document prefix are part of every
book's content hash, so changing either re-embeds every book on the next
sync. Anything else that searches the collection must embed its query with
the same model and query prefix.

## Scheduling and logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `POLL_INTERVAL_SECONDS` | `3600` | Time between syncs; `0` syncs once and exits (like `--once`) |
| `FORCE_REINGEST` | `false` | `true` re-embeds every book on the first sync after starting, ignoring change detection |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

## Compose (not read by the app)

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_VERSION` | `latest` | Image tag `docker-compose.prod.yaml` runs, e.g. `1.2.3` |

## Publishing (not read by the app)

| Variable | Meaning |
|----------|---------|
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
