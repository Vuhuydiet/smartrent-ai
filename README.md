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
├── domain/               # Domain layer (business entities)
│   ├── dtos              # Data tranfer objects
│   ├── entities/         # Database models
│   ├── repositories/     # Repositories
│   └── services/         # Domain services
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

### Local Development

1. **Clone the repository**
```bash
  git clone <repository-url>
  cd smartrent-ai
```

2. **Create virtual environment**
```bash
  python -m venv venv
  # Activate venv
	source venv/bin/activate # for Unix
	venv\\Scripts\\activate  # for Windows

```

3. **Install dependencies**
```bash
  pip install -r requirements.txt
```
Or easier, just run 
```bash
  make setup
```

4. **Set up environment variables**
```bash
  cp .env.example .env
  # Edit .env with your configuration
```

5. **Set up database**
```bash
  # Create MySQL database
  mysql -u root -p -e "CREATE DATABASE smartrent_ai;"
  mysql -u root -p -e "CREATE USER 'smartrent'@'localhost' IDENTIFIED BY 'password';"
  mysql -u root -p -e "GRANT ALL PRIVILEGES ON smartrent_ai.* TO 'smartrent'@'localhost';"
  mysql -u root -p -e "FLUSH PRIVILEGES;"
   
# Run migrations
  alembic upgrade head
```

6. **Run the application**
```bash
  uvicorn app.main:app --reload
```

## 🗄️ Database Migrations

This project uses Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Downgrade migrations
alembic downgrade -1
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_main.py
```

## 📚 API Documentation

Once the application is running, you can access:

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative API docs**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

## 🏗️ Development Guidelines

### Adding New Features

1. **Domain Layer**: Define entities in `app/domain/entities/`
2. **Repository**: Create implementation in `app/domain/repositories/`
4. **Dtos**: Define Pydantic models in `app/domain/dtos/`
5. **API**: Create endpoints in `app/api/v1/`
6. **Tests**: Add tests in the `tests/` directory

### Code Style

- Follow PEP 8 guidelines
- Use type hints throughout the codebase
- Write comprehensive docstrings
- Maintain test coverage above 80%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure tests pass: `pytest`
5. Commit your changes: `git commit -am 'Add new feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request
