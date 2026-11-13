# Instructions for AI assistants

Follow https://github.com/romkey/python-ai-rules/blob/main/RULES.md.
Follows python-ai-rules v0.1.0. When re-scanning, apply the entries in
its UPGRADING.md newer than that, then update this line.
With Claude Code, the `python-rules` plugin loads them for you.

Read docs/overview.md first. `main` is protected: work on a branch and
open a pull request; CI and Lint must pass before it can merge.

<!-- Project-specific notes go below. -->

- The catalog feed's shape comes from the library app's source
  (https://github.com/pdxhackerspace/library:
  `app/services/books/catalog_feed.rb`, `app/controllers/api/v1/`); check
  there rather than guessing.
- Keep `documents.py` pure (no I/O). Changing text or payload layout means
  bumping `DOCUMENT_SCHEMA_VERSION` so stored books are rebuilt.
- Never delete points in a cycle that didn't finish, or when the catalog
  came back empty.
- Deliberate differences from the rules: no `/healthz`, because the
  ingester listens on no port; the production health check is the
  `healthcheck` command reading a heartbeat file.
