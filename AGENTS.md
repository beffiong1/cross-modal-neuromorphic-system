# Repository Guidelines

## Project Structure & Module Organization
- `Models/` contains the core SNN architectures and memory mechanisms (baseline, SCL, Hopfield, HGRN, hybrid) plus shared layers/utilities.
- `Dataloaders/` provides dataset loaders (N-MNIST and SHD).
- `experiments/` contains dataset/model registries and a lightweight runner scaffold.
- `notebook/` holds the main experiment notebook (`cross_modal_experiments_complete (5).ipynb`).
- `results/` and `figures/` store generated metrics and plots.
- `training/` hosts reusable training pipeline utilities and loss functions.
- `info/` contains repo meta files (e.g., ignore lists).

## Build, Test, and Development Commands
There is no build system or CLI entrypoint in this repository. Typical workflows are:
- Run the main notebook for end-to-end experimentation: `jupyter notebook "notebook/cross_modal_experiments_complete (5).ipynb"`.
- Execute ad hoc scripts by importing modules in `Models/` and `Dataloaders/` from a Python session.

## Coding Style & Naming Conventions
- Follow PEP 8; keep functions/classes descriptive and aligned with the existing `Models/` naming (e.g., `model_2_scl.py`).
- Add docstrings to public functions/classes; type hints are encouraged.
- Use lowercase module filenames with underscores; keep new model files alongside existing ones in `Models/`.

## Testing Guidelines
- The repository references `pytest` usage and a `tests/` directory in `CONTRIBUTING.md`, but no tests are currently present.
- If you add tests, place them in `tests/` and run: `pytest tests/`.
- Aim for >80% coverage on new code where practical.

## Commit & Pull Request Guidelines
- Commit messages should be clear and concise; the suggested format is `Add feature: description`.
- Use feature branches (e.g., `feature/your-feature-name`).
- PRs should include a detailed description, link related issues, and note any performance changes; ensure tests pass before submission.

## Research/Analysis Notes
- New memory mechanisms or datasets should include a matching loader or model module, documentation updates, and (if added) tests.
- If results change, document comparisons in the PR description and update `results/` or `figures/` as needed.
