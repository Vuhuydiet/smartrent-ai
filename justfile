# SmartRent AI - Cross-platform task runner
# Install just: https://github.com/casey/just#installation
# Usage: just <command>

set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]
set shell := ["sh", "-cu"]

# Define virtual environment paths
venv_bin_dir := if os_family() == "windows" { justfile_directory() / "venv/Scripts" } else { justfile_directory() / "venv/bin" }
venv_python := venv_bin_dir / if os_family() == "windows" { "python.exe" } else { "python" }
venv_pip := venv_bin_dir / if os_family() == "windows" { "pip.exe" } else { "pip" }

# Default recipe to display available commands
default:
    just --list

# Setup project and install dependencies
install:
    python -m venv venv
    {{venv_pip}} install -r requirements.txt
    @echo "✅ Setup completed!"
    @echo "To activate the virtual environment:"
    @echo "  Windows: .\\venv\\Scripts\\activate"
    @echo "  Unix/macOS: source venv/bin/activate"

# Install pre-commit hooks
pre-commit-install:
    pre-commit install

# Run database migrations
migrate:
    alembic upgrade head

# Create new migration
migrate-create description:
    alembic revision --autogenerate -m "{{description}}"

# Downgrade last migration
migrate-downgrade:
    alembic downgrade -1

# Run tests
test:
    @echo "Running tests..."
    -pytest -v --cov=app --cov-report=html 2>/dev/null || pytest -v

# Run specific test file
test-file file:
    @echo "Running specific test file: {{file}}"
    -pytest {{file}} -v --cov=app --cov-report=html 2>/dev/null || pytest {{file}} -v

# Run development server
run:
    {{venv_python}} -m uvicorn app.main:app --reload --host localhost --port 8000

# Run development server (if venv is activated)
run-activated:
    uvicorn app.main:app --reload --host localhost --port 8000

# Run linting
lint:
    flake8 app tests
    mypy app

# Format code
format:
    black app tests
    isort app tests

# Check code formatting without making changes
format-check:
    black --check app tests
    isort --check-only app tests

# Run linting and fix issues automatically where possible
lint-fix:
    black app tests
    isort app tests
    flake8 app tests

# Run pre-commit hooks on all files
pre-commit-run:
    pre-commit run --all-files

# Run all code quality checks
check-all:
    @echo "Running all code quality checks..."
    black --check app tests
    isort --check-only app tests
    flake8 app tests
    mypy app
    @echo "All checks passed!"

# Clean up build artifacts (cross-platform Python solution)
clean:
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
    python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"
    python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyo')]"
    python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyd')]"
    python -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('.coverage')]"
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
    python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('htmlcov')]"

# Check project health
health:
    @echo "Checking project health..."
    python --version
    pip check
    just format-check
    just lint
