# SmartRent AI - Cross-platform task runner
# Install just: https://github.com/casey/just#installation
# Usage: just <command>

set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]
set shell := ["sh", "-cu"]

# Define virtual environment paths
venv_bin_dir := if os_family() == "windows" { justfile_directory() / "venv/Scripts" } else { justfile_directory() / "venv/bin" }
venv_python := venv_bin_dir / if os_family() == "windows" { "python.exe" } else { "python" }
venv_pip := venv_bin_dir / if os_family() == "windows" { "pip.exe" } else { "pip" }

# Cross-platform python launcher
python_launcher := if os_family() == "windows" { "py -3.10" } else { "python3" }

# Cross-platform venv creation command (prefer pyenv, then 3.10, then defaults)
create_venv_cmd := if os_family() == "windows" {
  "if (Get-Command pyenv -ErrorAction SilentlyContinue) { pyenv exec python -m venv venv } \n"
  + "elseif (Get-Command py -ErrorAction SilentlyContinue) { py -3.10 -m venv venv; if ($LASTEXITCODE -ne 0) { py -3 -m venv venv } } \n"
  + "else { python -m venv venv }"
} else {
  "if command -v pyenv >/dev/null 2>&1; then pyenv exec python -m venv venv; "
  + "elif command -v python3.10 >/dev/null 2>&1; then python3.10 -m venv venv; "
  + "elif command -v python3 >/dev/null 2>&1; then python3 -m venv venv; "
  + "else python -m venv venv; fi"
}

# Cross-platform pre-commit install command with hooksPath fallback
venv_pre_commit := venv_bin_dir / if os_family() == "windows" { "pre-commit.exe" } else { "pre-commit" }
pre_commit_install_cmd := if os_family() == "windows" {
  "$ErrorActionPreference = 'SilentlyContinue'; & '" + venv_pre_commit + "' install; if ($LASTEXITCODE -ne 0) { git config --unset-all core.hooksPath; git config --global --unset-all core.hooksPath; & '" + venv_pre_commit + "' install }"
} else {
  venv_pre_commit + " install || (git config --unset-all core.hooksPath || true; git config --global --unset-all core.hooksPath || true; " + venv_pre_commit + " install)"
}

# Default recipe to display available commands
default:
    just --list

# Setup project and install dependencies
install:
    {{create_venv_cmd}}
    {{venv_pip}} install -r requirements.txt
    @echo "✅ Setup completed!"
    @echo "To activate the virtual environment:"
    @echo "  Windows: .\\venv\\Scripts\\activate"
    @echo "  Unix/macOS: source venv/bin/activate"

# Install pre-commit hooks
pre-commit-install:
    {{pre_commit_install_cmd}}

# Run database migrations
migrate:
    {{venv_bin_dir}}/alembic upgrade head

# Create new migration
migrate-create description:
    {{venv_bin_dir}}/alembic revision --autogenerate -m "{{description}}"

# Downgrade last migration
migrate-downgrade:
    {{venv_bin_dir}}/alembic downgrade -1

# Run tests
test:
    @echo "Running tests..."
    -{{venv_bin_dir}}/pytest -v --cov=app --cov-report=html

# Run specific test file
test-file file:
    @echo "Running specific test file: {{file}}"
    -{{venv_bin_dir}}/pytest {{file}} -v --cov=app --cov-report=html

# Run development server
run:
    {{venv_python}} -m uvicorn app.main:app --reload --host localhost --port 8000

# Run development server (if venv is activated)
run-activated:
    {{venv_bin_dir}}/uvicorn app.main:app --reload --host localhost --port 8000

# Run linting
lint:
    {{venv_bin_dir}}/flake8 app tests
    {{venv_bin_dir}}/mypy .

# Format code
format:
    {{venv_bin_dir}}/black app tests
    {{venv_bin_dir}}/isort app tests

# Check code formatting without making changes
format-check:
    {{venv_bin_dir}}/black --check app tests
    {{venv_bin_dir}}/isort --check-only app tests

# Run linting and fix issues automatically where possible
lint-fix:
    {{venv_bin_dir}}/black app tests
    {{venv_bin_dir}}/isort app tests
    {{venv_bin_dir}}/flake8 app tests

# Run pre-commit hooks on all files
pre-commit-run:
    {{venv_bin_dir}}/pre-commit run --all-files

# Run all code quality checks
check-all:
    @echo "Running all code quality checks..."
    {{venv_bin_dir}}/black --check app tests
    {{venv_bin_dir}}/isort --check-only app tests
    {{venv_bin_dir}}/flake8 app tests
    {{venv_bin_dir}}/mypy .
    @echo "All checks passed!"

# Clean up build artifacts (cross-platform Python solution)
clean:
    {{venv_python}} -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
    {{venv_python}} -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"
    {{venv_python}} -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyo')]"
    {{venv_python}} -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyd')]"
    {{venv_python}} -c "import pathlib; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('.coverage')]"
    {{venv_python}} -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('*.egg-info')]"
    {{venv_python}} -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
    {{venv_python}} -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('htmlcov')]"

# Check project health
health:
    @echo "Checking project health..."
    {{venv_python}} --version
    {{venv_pip}} check
    just format-check
    just lint
