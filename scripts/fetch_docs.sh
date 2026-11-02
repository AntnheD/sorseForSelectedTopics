#!/usr/bin/env bash
# Reproducibly fetch the FastAPI documentation subset used as the corpus.
#
# Usage:  bash scripts/fetch_docs.sh [COMMIT]
# Result: data/documents/fastapi/docs/      Markdown pages (developer-facing sections)
#         data/documents/fastapi/docs_src/  code examples referenced by {* ... *} includes
#         data/documents/fastapi/SOURCE.md  provenance record
#
# Only the documented subset is copied (see INCLUDE_DIRS / INCLUDE_FILES).
set -euo pipefail

COMMIT="${1:-5cb6272ed7e3e07eaf6978e1e65b94aca76fa8e0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKOUT="$ROOT/.fastapi_checkout"
DEST="$ROOT/data/documents/fastapi"

INCLUDE_DIRS=(tutorial advanced deployment how-to)
INCLUDE_FILES=(index.md features.md async.md python-types.md environment-variables.md virtual-environments.md fastapi-cli.md _llm-test.md)

echo "== cloning fastapi/fastapi (sparse, docs only) at $COMMIT"
rm -rf "$CHECKOUT"
git clone --quiet --filter=blob:none --sparse https://github.com/fastapi/fastapi.git "$CHECKOUT"
git -C "$CHECKOUT" sparse-checkout set docs/en/docs docs_src >/dev/null
git -C "$CHECKOUT" checkout --quiet "$COMMIT"
DATE="$(git -C "$CHECKOUT" log -1 --format=%cs)"

echo "== copying documentation subset"
rm -rf "$DEST/docs" "$DEST/docs_src"
mkdir -p "$DEST/docs" "$DEST/docs_src"
for d in "${INCLUDE_DIRS[@]}"; do
  (cd "$CHECKOUT/docs/en/docs" && find "$d" -name '*.md' -exec cp --parents {} "$DEST/docs" \;)
done
for f in "${INCLUDE_FILES[@]}"; do
  cp "$CHECKOUT/docs/en/docs/$f" "$DEST/docs/$f"
done

echo "== copying referenced code examples"
grep -rhoE '\{(\*|!>?)\s*\.\./\.\./docs_src/[^ *!}]+' "$DEST/docs" \
  | sed -E 's/.*docs_src\///' | sort -u > "$DEST/.referenced_examples.txt"
while read -r rel; do
  src="$CHECKOUT/docs_src/$rel"
  if [ -f "$src" ]; then
    mkdir -p "$DEST/docs_src/$(dirname "$rel")"
    cp "$src" "$DEST/docs_src/$rel"
  fi
done < "$DEST/.referenced_examples.txt"
N_EX=$(wc -l < "$DEST/.referenced_examples.txt")
rm -f "$DEST/.referenced_examples.txt"

N_MD=$(find "$DEST/docs" -name '*.md' | wc -l)
N_PY=$(find "$DEST/docs_src" -type f | wc -l)

cat > "$DEST/SOURCE.md" <<EOF
# Corpus provenance

- Upstream: https://github.com/fastapi/fastapi (documentation is MIT licensed)
- Commit: \`$COMMIT\` (committed $DATE)
- Fetched with: \`bash scripts/fetch_docs.sh $COMMIT\`
- Markdown files copied: $N_MD (from \`docs/en/docs\`)
- Code example files copied: $N_PY of $N_EX referenced (from \`docs_src\`)

## Included
Directories: ${INCLUDE_DIRS[*]}
Top-level files: ${INCLUDE_FILES[*]}

## Excluded and why
- \`reference/\`: API reference stubs rendered by mkdocstrings (no prose in the Markdown).
- \`release-notes.md\`: changelog, version-specific noise that would dominate retrieval.
- \`about/\`, \`learn/\`, \`resources/\`, \`fastapi-people.md\`, \`newsletter.md\`, \`help-fastapi.md\`,
  \`contributing.md\`, \`management.md\`, \`translations.md\`, \`translation-banner.md\`,
  \`external-links.md\`, \`benchmarks.md\`, \`alternatives.md\`, \`history-design-future.md\`,
  \`project-generation.md\`, \`editor-support.md\`: community/meta pages, not developer usage docs.
- \`_llm-test.md\` is copied but excluded at preprocessing time (\`corpus.exclude_files\`): it is a
  synthetic translation-test page, not documentation.
- Images, CSS and JS are not copied.

## Caveats
- The docs describe the FastAPI version at this commit; behaviour may differ for older releases.
- Code examples are inlined at preprocessing time from \`docs_src\`; the docs' tabbed variants
  (Python 3.8/3.9/3.10) are all inlined, so some sections repeat similar code.
EOF

rm -rf "$CHECKOUT"
echo "== done: $N_MD Markdown files, $N_PY code examples -> $DEST"
