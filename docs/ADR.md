# Architectural Decision Records (ADR)

## ADR-001: Retention of Raw Dataset in Early Git History
- **Status**: Accepted
- **Date**: 2026-11-16
- **Context**: The raw Excel dataset (`data/raws/online_retail_II.xlsx`, 43.51 MiB) was committed in the repository's initial commit (`5d4ab69`) and already exists on `origin/main`.
- **Decision**: Keep history as-is. Do not rewrite history and do not force-push to `main`.
- **Rationale**:
  1. The file exists in early git history (`5d4ab69` through `3304215`), but has been completely untracked from the working tree since commit `d32f64a`.
  2. The raw dataset is strictly ignored going forward via [.gitignore](.gitignore) (`data/raws/*` with exception for `data/raws/.gitkeep`).
  3. Total repository size is 47.77 MiB, well below GitHub's 100 MB per-file and 2 GB repository limits.
  4. Preserves cryptographic and hash stability across all existing commits and references.
