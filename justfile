# https://just.systems

# list available recipes
default:
    @just --list

# install the dev environment and pre-commit hooks
install:
    uv sync
    uv run pre-commit install

# run tests with pytest
test:
    uv run pytest

# run tests with pytest and show coverage
test-cov:
    uv run pytest --cov --cov-report=term-missing

# run linting and formatting on all files
lint:
    uv run pre-commit run --all-files

# run type checking
typecheck:
    uv run mypy

# upgrade locked dependency versions (commit the resulting uv.lock)
upgrade:
    uv lock --upgrade

# upgrade a single locked dependency, e.g. `just upgrade-package numpy`
upgrade-package package:
    uv lock --upgrade-package {{ package }}

# serve a live, sortable report of every run, e.g. `just report scripts/configs/p4_config.toml`
report config:
    uv run --extra report panel serve scripts/open_report.py --show --args {{ config }}

# view one p4 run's raw volume + masks in neuroglancer, e.g. `just view-p4 scripts/configs/p4_config.toml beneficial-dragon`
view-p4 config slug:
    uv run --extra view scripts/serve_p4_view.py {{ config }} {{ slug }}

# build wheel and sdist
build:
    uv build

# build docs and start a local server to preview them
docs-serve:
    uv run --group docs --isolated --no-dev mkdocs serve

# build docs in strict mode
docs-build:
    uv run --group docs --isolated --no-dev mkdocs build --strict

# tag and release <version>
release version:
    git tag -a {{ version }} -m {{ version }}
    git push origin --follow-tags
