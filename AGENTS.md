# Contributor Guidelines for Stock-App1

This repository hosts ML classifier (trees and logistic classifier).  Follow these rules when adding or updating code.

# Important — Not authoritative:
1. This file is only a quick-start contributor guide.
2. Your job is to adapt google colab python code to Github.


## Coding Standards
- One domain concept per file; no cyclic imports.
- Functions should stay under 20 lines with at most 2 nesting levels.
- Favour composition over inheritance and keep variables scoped tightly.
- Validate inputs early and throw on bad data.
- Use 2‑space indentation, single quotes and end files with a newline.
- Document each public API/function with a doc comment.


## Testing & CI
- Run tests before committing.

## Contributing Workflow
- **Fork** then branch off `main` using the pattern `feat/<topic>`.
- **Ensure local tests pass** before opening a PR.
- **Each PR requires at least one reviewer.**

to understand the current stage, past decisions, and open questions tied to the spec.

Refer to `README.md` and full documentation in `docs/` for further details on features and architecture.
