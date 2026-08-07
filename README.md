# cellpose-runner

[![CI](https://github.com/JaneliaSciComp/cellpose_runner/actions/workflows/ci.yaml/badge.svg)](https://github.com/JaneliaSciComp/cellpose_runner/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/JaneliaSciComp/cellpose_runner/branch/main/graph/badge.svg)](https://codecov.io/gh/JaneliaSciComp/cellpose_runner)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://JaneliaSciComp.github.io/cellpose_runner/)

Run CellPose with formalized configuration logging and visualization.

## Installation

```bash
pip install git+https://github.com/JaneliaSciComp/cellpose_runner
```

## Development

This project uses [uv](https://docs.astral.sh/uv/) and
[just](https://just.systems). To set up a development environment:

```bash
git clone https://github.com/JaneliaSciComp/cellpose_runner
cd cellpose_runner
just install
```

See [CONTRIBUTING.md](https://github.com/JaneliaSciComp/cellpose_runner/blob/main/CONTRIBUTING.md)
for more, or run `just` to list all available recipes.
