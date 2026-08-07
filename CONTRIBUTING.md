# Contributing to cellpose-runner

## Setting up a development environment

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and
[just](https://just.systems) as a task runner. Install both first:

- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
- [just installation](https://just.systems/man/en/packages.html)

`just` is only a convenience wrapper — every recipe is a one-line `uv run ...`
command you can copy out of the `justfile` and run directly.

```bash
git clone https://github.com/JaneliaSciComp/cellpose_runner
cd cellpose_runner
just install
```

`just install` creates the virtual environment (`uv sync`) and installs the
pre-commit hooks. Run `just` to see all available recipes.

## Dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`, which is
committed. The lock file is what makes an environment reproducible months later
and keeps local and cluster runs identical; it resolves for every platform and
Python version allowed by `requires-python`, so a lock file created on one OS
works on the others.

Add or remove dependencies with uv, which updates `pyproject.toml` and the lock
file together:

```bash
uv add numpy
uv remove numpy
```

Because versions are pinned, new upstream releases are *not* picked up
automatically. Upgrading is a deliberate step:

```bash
just upgrade                  # upgrade everything within the declared constraints
just upgrade-package numpy    # upgrade a single dependency
```

Commit the resulting `uv.lock` alongside the change. Outside of these commands
the lock file stays put — routine `just test` / `uv run` only read it.

These commands only move versions *within* the constraints in `pyproject.toml`;
they never rewrite the constraints themselves. A dependency declared as
`foo>=1,<2` will stay on 1.x however often you upgrade. Moving to 2.x means
editing the bound in `pyproject.toml` by hand, then re-locking — which is the
point at which you find out whether the new major breaks anything.

Dependabot opens a monthly pull request bumping `uv.lock`, and CI runs the full
test matrix against the new versions, so most upgrades arrive as a PR you only
need to review and merge.

Releases that fall *outside* the declared bounds never reach those PRs, so a
monthly `Outdated dependencies` workflow lists them in a GitHub issue instead.
It only reports; acting on it is the manual bound-widening step above.

## Tests, linting and type checking

```bash
just test        # run the test suite
just test-cov    # run the test suite with a coverage report
just lint        # run ruff (lint + format) and the other pre-commit hooks
just typecheck   # run mypy
```

Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/) via
pre-commit, so most style issues are fixed automatically when you commit.

## Documentation

Documentation is built with [mkdocs-material](https://squidfunk.github.io/mkdocs-material/)
and API pages are generated from docstrings (Google style).

```bash
just docs-serve  # preview the docs locally with live reload
just docs-build  # build the docs in strict mode, as CI does
```

## Pull requests

CI runs the test suite across Linux, macOS and Windows, type checks with mypy,
and builds the docs. Please make sure `just test`, `just lint` and
`just typecheck` pass locally before opening a pull request.

## Releasing

Releases are triggered by pushing a version tag. `setuptools-scm` derives the
version from the tag, and CI creates a GitHub release.

```bash
just release v0.1.0
```

