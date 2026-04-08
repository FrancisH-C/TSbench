# Contributing to TSbench

Thank you for your interest in contributing to TSbench! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/TSbench.git
   cd TSbench
   ```

2. Install in development mode with test dependencies:
   ```bash
   python -m pip install -e .[test]
   ```

3. Create a branch for your changes:
   ```bash
   git checkout -b my-feature
   ```

## Development Workflow

### Running Tests

```bash
python -m pytest -x -s              # basic tests
python -m pytest --run-R            # include R-dependent tests (requires R)
python -m pytest --run-all          # all tests
python -m pytest --cov=TSbench      # with coverage report
```

### Linting

```bash
ruff check .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

### Building Documentation

```bash
cd docs && make clean && make html
```

The built site will be in `docs/build/html/`.

## Submitting Changes

1. Make sure all tests pass before submitting.
2. Open a pull request against the `dev` branch with a clear description of the changes.
3. Keep pull requests focused on a single change.

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/FrancisH-C/TSbench/issues) to report bugs or request features. When reporting a bug, please include:

- A minimal reproducible example
- Python version and OS
- The full error traceback

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
