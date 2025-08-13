.PHONY: help install test lint format run clean docker-build docker-run migrate

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

test: ## Run tests
	@echo "🧪 Running tests..."
	pytest -v --cov=app --cov-report=html

lint: ## Run linting
	@echo "🔍 Running linting..."
	flake8 app tests
	mypy app

format: ## Format code
	@echo "🎨 Formatting code..."
	black app tests
	isort app tests

run: ## Run development server
	@echo "🚀 Starting development server..."
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

clean: ## Clean up build artifacts
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +

migrate: ## Run database migrations
	@echo "🗄️ Running database migrations..."
	alembic upgrade head

migrate-create: ## Create new migration
	@echo "🗄️ Creating new migration..."
	@read -p "Migration description: " desc; \
	alembic revision --autogenerate -m "$$desc"

setup: ## Initial project setup
	@echo "⚙️ Setting up project..."
	python -m venv venv
	@echo "Please activate virtual environment and run 'make install'"
	@echo "On Unix: source venv/bin/activate"
	@echo "On Windows: venv\\Scripts\\activ

