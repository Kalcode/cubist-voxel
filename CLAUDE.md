# Cubist

Agent-native voxel asset authoring. `docs/design.md` is the design of record; read it before
changing behavior, and update it when a decision changes.

## Commands

- `uv run pytest` — full suite (must stay green)
- `uv run ruff check . && uv run ruff format .`
- `uv run cubist build projects/demo/assets/<name>.py`, `uv run cubist render <out dir> --slices y=0..3`
- `CUBIST_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` after an intentional rendering
  change; look at every regenerated PNG in `tests/golden/` before committing it

## Conventions

- Tests first: write the failing test, watch it fail, then implement.
- Y-up, right-handed, +Z front; a voxel is filled when its center is inside or on a shape.
- Asset scripts use `from cubist import *` and reference colors by palette name only.
- Errors raised to scripts are written for an agent to act on: say what is wrong and how to fix it.
- Track milestone status in `docs/design.md` under Build plan.
