# Repository Guidelines

## Existing Project Description
- The existing repository is the repository taht contains experiments which support the paper: Modality_Dependent_Memory_Mechanisms_in_Cross_Modal_Neuromorphic_Computing_2026 (1).pdf
- The existing implementation benchmars 5 models (Models/model_1_baseline.py, Models/model_2_scl.py, Models/model_3_hopfield.py, Models/model_4_hgrn.py and Models/model_5_hybrid.py) against dataset: N-MNIST and SHD. The driver code is notebook/cross_modal_experiments_complete (5).ipynb

## Goal
- You will help implement new driver code that supports new experiments: Run the existing models against new datasets: https://tonic.readthedocs.io/en/latest/datasets.html for ablation study
- You may need to write new Dataloaders (Dataloaders) to support new models
- You may need to refactor the existing code base to improve modularization or adapt to new experiments

## Project Structure & Module Organization
- `Models/` contains the core SNN architectures and memory mechanisms (baseline, SCL, Hopfield, HGRN, hybrid) plus shared layers/utilities.
- `Dataloaders/` provides dataset loaders (N-MNIST and SHD).
- `notebook/` holds the main experiment notebook (`cross_modal_experiments_complete (5).ipynb`).
- `results/` and `figures/` store generated metrics and plots.
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

## Running environment
- This code base is deployed to a Runpod pod.
- runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

