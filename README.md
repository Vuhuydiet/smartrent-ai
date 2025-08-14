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
├── dtos                  # Data tranfer objects
├── entities/             # Database models
├── repositories/         # Repositories
└── services/             # Domain services
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

- Python 3.11+
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

| Command              | Description                                          |
| -------------------- | ---------------------------------------------------- |
| `install`            | Install/update dependencies                          |
| `test`               | Run tests                                            |
| `lint`               | Run linting                                          |
| `format`             | Format code                                          |
| `format-check`       | Check formatting without changes                     |
| `lint-fix`           | Run linting with auto-fix                            |
| `pre-commit-install` | Install pre-commit hooks                             |
| `pre-commit-run`     | Run pre-commit hooks                                 |
| `check-all`          | Run all code quality checks                          |
| `run`                | Start development server                             |
| `clean`              | Clean up build artifacts                             |
| `migrate`            | Run database migrations                              |
| `dev-setup`          | Full development environment setup                   |
| `health`             | Check project health                                 |

### Local Development

1. **Clone the repository**

```bash
  git clone <repository-url>
  cd smartrent-ai
```

2. **Setup project (creates venv and installs dependencies)**

```bash
  just install-for-<os>
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

### Adding New Features

1. **Domain Layer**: Define entities in `app/entities/`
2. **Repository**: Create implementation in `app/repositories/`
3. **Dtos**: Define Pydantic models in `app/dtos/`
4. **Services**: Define business logic in `app/services`
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

#### VS Code Integration

The project includes VS Code settings (`.vscode/settings.json`) that:

- Automatically format code on save
- Organize imports on save
- Enable linting and type checking
- Set the correct Python interpreter

#### Code Quality Guidelines

- Follow PEP 8 guidelines (enforced by Flake8)
- Use type hints throughout the codebase (checked by MyPy)
- Write comprehensive docstrings
- Maintain test coverage above 80%
- All code must pass linting and formatting checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request
