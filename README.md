# Library RAG ingester

[![CI](https://github.com/romkey/library-rag-ingester/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/library-rag-ingester/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/library-rag-ingester/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/library-rag-ingester/actions/workflows/lint.yaml)
[![Python 3.15](https://img.shields.io/badge/python-3.15-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Keeps a [Qdrant](https://qdrant.tech/) collection in step with the
[PDX Hackerspace library](https://github.com/pdxhackerspace/library)
catalog, so a retrieval-augmented chatbot can answer "do we have a book on
soldering?" or "what's on shelf B2?". It reads the library's catalog feed
with an API key, embeds each book with an OpenAI-compatible embeddings API,
and re-syncs on a schedule, embedding only what changed and removing books
that left the catalog. It works like the other PDX Hackerspace RAG
ingesters ([audiobookshelf](https://github.com/romkey/audiobookshelf-rag-ingester),
[events](https://github.com/romkey/pdxhackerspace-events2rag),
[wiki](https://github.com/romkey/wikijs2rag)).

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/library_rag_ingester/__main__.py` | Command line: sync, `query`, `healthcheck`; logging and SIGTERM |
| `src/library_rag_ingester/config.py` | Settings from environment variables |
| `src/library_rag_ingester/catalog.py` | Fetches and parses the library's catalog feed |
| `src/library_rag_ingester/documents.py` | Turns a book into embedding text, payload and content hash |
| `src/library_rag_ingester/embeddings.py` | OpenAI-compatible embeddings client, with prompt prefixes |
| `src/library_rag_ingester/qdrant_store.py` | Qdrant collection, payload indexes, upserts, deletes, search |
| `src/library_rag_ingester/service.py` | The sync cycle and polling loop |
| `src/library_rag_ingester/heartbeat.py`, `query.py` | Health-check heartbeat; search output |
| `tests/` | pytest tests; Qdrant runs in memory and HTTP is faked |
| `docs/configuration.md` | Every environment variable |
| `docs/overview.md` | Short map of the code and its invariants |

## Technical details

**Source.** The library app serves its catalog at `GET /api/v1/catalog` as
[JSON Lines](https://jsonlines.org/), one book per line, to clients that
send `Authorization: Bearer <API key>`. An admin creates a key in the
library under **Settings → API keys**. Each record has the book's id, URL,
title, subtitle, authors, subjects, ISBNs, publisher, publication date,
language, page count, shelf location, copy count, description, notes,
cover and ebook URLs, when it was added and updated, and a ready-to-embed
`text` field. Loans and borrowers are never in it. The ingester also
accepts a plain JSON array of the same records.

**Sync.** Each cycle:

1. Fetches the catalog, sending the previous `ETag` as `If-None-Match`.
   A `304 Not Modified` ends the cycle.
2. Builds one document per book: the feed's `text` (or the same layout
   built from the fields) and a payload of the catalog fields.
3. Compares each book's content hash (text, payload, model, document
   prefix) with the one stored in Qdrant, and embeds and upserts only the
   new and changed books.
4. Deletes stored books that are no longer in the catalog. The feed
   doesn't report deletions, so every sync fetches the whole catalog
   rather than using `?since=`. An empty catalog, or a sync stopped by
   SIGTERM, deletes nothing.

**Qdrant points.** One per book; the id is a UUIDv5 of the book id. The
payload has `source: "library"`, `chunk_type: "book"`, `book_id`, `url`,
`title`, `subtitle`, `full_title`, `authors`, `subjects`, `isbns`,
`publisher`, `published_on`, `published_year`, `language`, `page_count`,
`location`, `copies_count`, `description`, `notes`, `ebook_url`,
`cover_url`, `metadata_source`, `source_url`, `added_on`, `added_at`,
`updated_at`, `text`, `content_hash` and `ingested_at`. `source`,
`chunk_type`, `book_id`, `authors`, `subjects`, `isbns`, `location`,
`language` and `published_year` have keyword indexes, `added_at` and
`updated_at` datetime indexes.

**Searching the collection.** Embed the query with the same model and the
query prefix (`search_query: ` by default), and filter on
`source = "library"` if the collection is shared. Or use the built-in
command:

```bash
library-rag-ingester query "intro to soldering"
library-rag-ingester query --limit 10 --min-score 0.5 --show-text "machine learning"
```

**Command line.**

| Command | What it does |
|---------|--------------|
| `library-rag-ingester` | Sync now and every `POLL_INTERVAL_SECONDS` until SIGTERM |
| `library-rag-ingester --once` | Sync once and exit (status 1 if it failed) |
| `library-rag-ingester query TEXT` | Search the collection |
| `library-rag-ingester healthcheck` | Exit 0 if the running sync touched its heartbeat file in the last 10 minutes |
| `library-rag-ingester --version` | Print the version |

## Running it

Requires Docker and an OpenAI-compatible embeddings server (or an OpenAI
API key). Configuration comes from environment variables, kept in `.env`
(see [docs/configuration.md](docs/configuration.md)); at least set
`LIBRARY_CATALOG_URL` and `LIBRARY_API_KEY`:

```bash
cp .env.example .env   # then edit it
docker compose -f docker-compose.prod.yaml up -d
```

That runs the latest release from `ghcr.io/romkey/library-rag-ingester`
with its own Qdrant (on `127.0.0.1:6333`); set `APP_VERSION=X.Y.Z` to pin a
release. To use a Qdrant you already run, delete the `qdrant` service from
the compose file and set `QDRANT_URL`.

The default model, `nomic-ai/modernbert-embed-base`, can be served by
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
on the Docker host:

```bash
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest --model-id nomic-ai/modernbert-embed-base
```

### With pip

Needs Python 3.15 or newer. Install into a virtual environment, from a
release's wheel:

```bash
python3 -m venv ~/.venvs/library-rag-ingester
~/.venvs/library-rag-ingester/bin/pip install https://github.com/romkey/library-rag-ingester/releases/download/vX.Y.Z/library_rag_ingester-X.Y.Z-py3-none-any.whl
~/.venvs/library-rag-ingester/bin/library-rag-ingester --version
```

Set its environment variables yourself (see
[docs/configuration.md](docs/configuration.md)); it doesn't read `.env`.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # run from the working tree, with Qdrant
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run library-rag-ingester --once
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
library-rag-ingester --once
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit.

## License

[MIT](LICENSE)
