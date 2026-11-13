# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- The release workflow no longer fails saving the Docker build cache when
  a merged release PR starts it.

## [0.1.0] - 2026-11-13

### Added

- Syncs the PDX Hackerspace library catalog (`/api/v1/catalog`, JSON Lines,
  Bearer API key) into Qdrant, one point per book, embedded with an
  OpenAI-compatible API.
- Change detection by content hash and ETag; books removed from the catalog
  are deleted from the collection.
- `query` command for semantic search, `healthcheck` command and heartbeat
  file for the production health check, `--once` for a single sync.
