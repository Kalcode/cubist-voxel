# Contributing to Cubist

Thanks for wanting to help. Cubist has one maintainer, so every contribution starts as a
conversation: first a discussion, then a pull request if the change is wanted.

## How a change gets in

1. **Start a [discussion](https://github.com/Kalcode/cubist-voxel/discussions).** Bugs, ideas,
   questions and proposed changes all begin there. Describe the problem before the fix: what
   you were trying to make, what happened, and the smallest script that shows it.
2. **Agree on the change.** When the maintainer agrees something should change, the
   discussion is marked answered or turned into an issue, with the approach settled.
3. **Then open a pull request** that links to it. Pull requests without an agreed discussion
   are closed with a pointer back here.

Typo and broken-link fixes can skip straight to a pull request.

Why this order: writing a change is cheap now, and reviewing one is not. Agreeing first means
nobody spends an evening on a pull request that can't be merged, and review time goes to
changes people asked for. It is not meant to keep anyone out. Discussions are where most of
the useful work happens anyway: a clear bug report with a failing script is worth a lot.

## Using AI tools

Cubist is built with AI agents, and it exists so agents can make game assets, so AI-assisted
contributions are welcome. They follow the same process, plus:

- **You're the author.** You understand every line, and you've run the tests yourself.
- **Say what you used**, in the pull request: which tools, for which parts. "None" is a fine
  answer.
- **One focused change per pull request**, tied to its discussion. Bulk, unsolicited or
  drive-by generated pull requests are closed without review.

If you point an agent at this repo, it will find these rules in `CLAUDE.md` and `AGENTS.md`.

With Claude Code, the project turns on the [superpowers](https://github.com/obra/superpowers)
plugin (`.claude/settings.json`). It brainstorms a change with you, writes a plan, and builds
it test-first, which suits how changes are made here, and its brainstorming fits the discussion
step. The specs and plans it writes go in `docs/superpowers/`, which git ignores, so they stay
on your machine.

## Setting up

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Kalcode/cubist-voxel.git
cd cubist-voxel
uv sync
uv run pytest
```

Some tests need extra tools and skip when they're missing: [Godot 4](https://godotengine.org/)
(`godot` on `PATH`) checks GLB imports, and Blender 5 checks animation poses.

## Making the change

- **Write the test first:** watch it fail, then make it pass.
- **Run the checks** before pushing:
  `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
- **Rendering changes:** regenerate the golden images with
  `CUBIST_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py`, look at every changed PNG in
  `tests/golden/`, and say in the pull request why each one changed.
- **Update the docs your change touches.** `docs/design.md` is the design of record, so update
  it when a decision changes. `docs/dsl.md` covers anything scripts can call, and
  `docs/manifest.md` covers the engine contract.
- **Manifest changes:** new optional fields keep `cubist_manifest` at version 1. Changing what
  an existing field means bumps the version, and that needs its own discussion.
- **Write errors for agents.** An error raised to a script or tool says what is wrong and how to
  fix it.
- **Follow the conventions in [`CLAUDE.md`](CLAUDE.md):** Y up, right-handed, +Z is the front;
  scripts use palette names, never hex.

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).

GitHub hosts a mirror of the main repository. An accepted pull request is applied there and
shows up here on the next sync, so GitHub may mark it closed rather than merged. Your commits
keep your name either way.

## Elsewhere

- Security problems: see [SECURITY.md](SECURITY.md). Please don't post them in public.
- Conduct: everyone here follows the [Code of Conduct](CODE_OF_CONDUCT.md).
- Made something with Cubist? Share it under Show and tell in the discussions.
