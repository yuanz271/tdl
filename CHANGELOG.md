# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2025-10-10

### Added
- Documented Ruff linting and formatting commands in `AGENTS.md`.
- Captured recent automation outputs in `changes.log`.
- Exposed the package as `tdl` with a console entry point `run-sarsa`.
- Provided an `examples` optional dependency that includes JupyterLab for notebook workflows.

### Changed
- Applied `uvx ruff format` across the repository to enforce consistent style.
- Converted the source tree to a `src/tdl` package layout for distribution.
- Relocated the SARSA runner and experiment helpers into a top-level `examples/` directory to keep algorithms standalone.
- Removed the package-level entry point and captured orchestration inside `examples/sarsa.ipynb` as an interactive example.
- Kept experiment helpers next to the walkthrough so the `tdl` package remains task-agnostic.
- Compute stepwise rewards during `run` so `update` can consume them directly, keeping reward-related parameters consistent across trajectories.
