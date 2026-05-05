# SmartRent AI - Cross-platform task runner
# Install just: https://github.com/casey/just#installation
# Usage: just <command>

set windows-shell := ["pwsh.exe", "-NoLogo", "-Command"]
set shell := ["sh", "-cu"]

# Default recipe to display available commands
default:
    just --list

# Setup project and install dependencies with uv
install:
    uv sync
    @echo "✅ Setup completed!"
    @echo "Run commands with: uv run <command>"
    @echo "Or activate the environment: source .venv/bin/activate (Unix) or .venv\\Scripts\\activate (Windows)"

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install

# Run database migrations
migrate:
    uv run alembic upgrade head

# Create new migration
migrate-create description:
    uv run alembic revision --autogenerate -m "{{description}}"

# Downgrade last migration
migrate-downgrade:
    uv run alembic downgrade -1

# Run tests
test:
    @echo "Running tests..."
    uv run pytest -v --cov=app --cov-report=html

# Run specific test file
test-file file:
    @echo "Running specific test file: {{file}}"
    uv run pytest {{file}} -v --cov=app --cov-report=html

# Run development server
run:
    uv run uvicorn app.main:app --reload --host localhost --port 8000

# Run linting
lint:
    uv run flake8 app tests
    uv run mypy .

# Format code
format:
    uv run black app tests
    uv run isort app tests

# Check code formatting without making changes
format-check:
    uv run black --check app tests
    uv run isort --check-only app tests

# Run linting and fix issues automatically where possible
lint-fix:
    uv run black app tests
    uv run isort app tests
    uv run flake8 app tests

# Run pre-commit hooks on all files
pre-commit-run:
    uv run pre-commit run --all-files

# Run all code quality checks
check-all:
    @echo "Running all code quality checks..."
    uv run black --check app tests
    uv run isort --check-only app tests
    uv run flake8 app tests
    uv run mypy .
    @echo "All checks passed!"

# Clean up build artifacts
clean:
    rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov tests/htmlcov tests/.coverage
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Check project health
health:
    @echo "Checking project health..."
    uv run python --version
    uv pip check
    just format-check
    just lint

# Add new dependency
add-dep package:
    uv add {{package}}

# Add new dev dependency
add-dev-dep package:
    uv add --dev {{package}}

# Update all dependencies
update-deps:
    uv sync --upgrade

# Lock dependencies without installing
lock:
    uv lock
