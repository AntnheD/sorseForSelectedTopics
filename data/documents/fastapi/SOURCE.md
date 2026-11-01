# Corpus provenance

- Upstream: https://github.com/fastapi/fastapi (documentation is MIT licensed)
- Commit: `5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0` (committed 2026-11-02)
- Fetched with: `bash scripts/fetch_docs.sh 5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0`
- Markdown files copied: 115 (from `docs/en/docs`)
- Code example files copied: 303 of 303 referenced (from `docs_src`)

## Included
Directories: tutorial advanced deployment how-to
Top-level files: index.md features.md async.md python-types.md environment-variables.md virtual-environments.md fastapi-cli.md _llm-test.md

## Excluded and why
- `reference/`: API reference stubs rendered by mkdocstrings (no prose in the Markdown).
- `release-notes.md`: changelog, version-specific noise that would dominate retrieval.
- `about/`, `learn/`, `resources/`, `fastapi-people.md`, `newsletter.md`, `help-fastapi.md`,
  `contributing.md`, `management.md`, `translations.md`, `translation-banner.md`,
  `external-links.md`, `benchmarks.md`, `alternatives.md`, `history-design-future.md`,
  `project-generation.md`, `editor-support.md`: community/meta pages, not developer usage docs.
- `_llm-test.md` is copied but excluded at preprocessing time (`corpus.exclude_files`): it is a
  synthetic translation-test page, not documentation.
- Images, CSS and JS are not copied.

## Caveats
- The docs describe the FastAPI version at this commit; behaviour may differ for older releases.
- Code examples are inlined at preprocessing time from `docs_src`; the docs' tabbed variants
  (Python 3.8/3.9/3.10) are all inlined, so some sections repeat similar code.
