# SmartRent AI

A FastAPI application built with Clean Architecture principles for the SmartRent AI project.

## 🏗️ Architecture

This project follows Clean Architecture principles with clear separation of concerns:

```
app/
├── api/                  # API layer (controllers)
│   └── v1/
│       └── api.py        # API router configuration
├── core/                 # Core configuration and utilities
│   ├── config.py         # Application settings
├── database/             # Database configuration
│   └── database.py       # Database connection
├── ai/                   # Core AI features
├── dto                   # Data tranfer objects
├── entitiy/              # Database models
├── repository/           # Repositories
└── service/              # Domain services
```

## 🚀 Features

- **Clean Architecture**: Organized in layers with clear dependencies
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: Powerful ORM for database operations
- **Alembic**: Database migration management
- **Pydantic**: Data validation using Python type annotations
- **Docker Support**: Containerized application with Docker
- **Test Suite**: Comprehensive testing with pytest
- **Type Hints**: Full type annotation support

## 📋 Requirements

- Python 3.10+
- MySQL 8.0+
- Docker (optional)

## 🛠️ Installation

### Task Runner

This project uses **Just** as a cross-platform task runner.

#### **Just** - `justfile`

Modern command runner with excellent cross-platform support.

**Installation:**

```bash
# Windows (via Chocolatey)
choco install just

# macOS
brew install just

# Linux
cargo install just
```

#### Available Commands

| Command              | Description                      |
| -------------------- | -------------------------------- |
| `install`            | Install/update dependencies      |
| `test`               | Run tests                        |
| `test-file <file>`   | Run specific test file           |
| `lint`               | Run linting                      |
| `format`             | Format code                      |
| `format-check`       | Check formatting without changes |
| `lint-fix`           | Run linting with auto-fix        |
| `pre-commit-install` | Install pre-commit hooks         |
| `pre-commit-run`     | Run pre-commit hooks             |
| `check-all`          | Run all code quality checks      |
| `run`                | Start development server         |
| `run-activated`      | Start server (if venv activated) |
| `clean`              | Clean up build artifacts         |
| `migrate`            | Run database migrations          |
| `migrate-create`     | Create new migration             |
| `migrate-downgrade`  | Downgrade last migration         |
| `health`             | Check project health             |

### Local Development

1. **Clone the repository**

```bash
  git clone <repository-url>
  cd smartrent-ai
```

2. **Setup project (creates venv and installs dependencies)**

```bash
  just install
  just pre-commit-install
```

3. **Set up environment variables**

```bash
  cp .env.example .env
  # Edit .env with your configuration
```

4. **Set up database**

```bash
  # Create MySQL database
  mysql -u root -p -e "CREATE DATABASE smartrent_ai;"
  mysql -u root -p -e "CREATE USER 'smartrent'@'localhost' IDENTIFIED BY 'password';"
  mysql -u root -p -e "GRANT ALL PRIVILEGES ON smartrent_ai.* TO 'smartrent'@'localhost';"
  mysql -u root -p -e "FLUSH PRIVILEGES;"

  # Run migrations
  just migrate
```

6. **Run the application**

```bash
  just run
```

## 🗄️ Database Migrations

This project uses Alembic for database migrations:

```bash
# Create a new migration
just migrate-create <description>

# Apply migrations
just migrate

# Downgrade migrations
just migrate-downgrade
```

## 🧪 Testing

Run the test suite:

```bash
# Run with coverage
just test

# Run specific test file
just test-file <file>
```

## 📚 API Documentation

Once the application is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative API docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

## 🏗️ Development Guidelines

### CI/CD Pipeline

This project includes a comprehensive GitHub Actions CI/CD pipeline that:

- **Tests**: Runs unit tests, code quality checks, and coverage analysis
- **Security**: Scans Docker images for vulnerabilities
- **Build & Deploy**: Creates and pushes Docker images to DockerHub

#### Pipeline Triggers

- Push to `main` or `dev` branches
- Pull requests to `main` branch

#### Required Secrets

Set these in your GitHub repository settings:

- `DOCKERHUB_USERNAME`: Your DockerHub username
- `DOCKERHUB_TOKEN`: DockerHub access token

📖 **Detailed Setup Guide**: See [CI/CD Setup Documentation](docs/ci-cd-setup.md)

### Adding New Features

1. **Domain Layer**: Define entities in `app/entity/`
2. **Repository**: Create implementation in `app/repository/`
3. **Dtos**: Define Pydantic models in `app/dto/`
4. **Services**: Define business logic in `app/service`
5. **API**: Create endpoints in `app/api/v1/`
6. **AI**: AI features in `app/ai/` directory
7. **Tests**: Add tests in the `tests/` directory

### Code Style

This project uses several tools to maintain code quality and consistency:

#### Development Tools

- **Black**: Code formatter that enforces a consistent style
- **isort**: Import sorter that organizes imports according to PEP 8
- **Flake8**: Linter that checks for code quality and style issues
- **MyPy**: Static type checker for Python
- **Pre-commit**: Git hooks that run checks before commits

#### Configuration Files

- `pyproject.toml`: Configuration for Black, isort, MyPy, and pytest
- `.flake8`: Flake8 configuration
- `.pre-commit-config.yaml`: Pre-commit hooks configuration

#### Code Quality Guidelines

- Follow PEP 8 guidelines (enforced by Flake8)
- Use type hints throughout the codebase (checked by MyPy)
- Write comprehensive docstrings
- Maintain test coverage above 80%
- All code must pass linting and formatting checks
