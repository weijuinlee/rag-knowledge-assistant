# Contributing

Thanks for contributing.

## Suggested workflow

1. Create a branch for your change.
2. Add tests for behavior changes in `tests/`.
3. Keep API changes backward-compatible where possible.
4. Run `pytest` before opening a PR.
5. Update `CHANGELOG.md` with user-facing changes.
6. Update `ROADMAP.md` when planning follow-up work.

## Design principles

- Keep retrieval deterministic and easy to debug.
- Prefer small, testable functions over hidden logic.
- Preserve compatibility with existing request/response schemas when possible.

## Code style

- 4-space indentation
- Use type hints for all new public functions
- Keep functions focused and explicit
- Prefer environment-first configuration (see `src/rag_assistant/config.py`).
- Keep security controls explicit by defaulting to disabled and documenting expected deployments.
