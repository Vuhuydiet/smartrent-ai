# CI/CD Setup Guide

This document explains how to set up the GitHub Actions CI/CD pipeline for the SmartRent AI project.

## Overview

The CI/CD pipeline includes:

- **Testing**: Runs unit tests, code quality checks, and coverage analysis
- **Building**: Creates Docker images for multiple architectures
- **Security**: Scans Docker images for vulnerabilities
- **Deployment**: Pushes images to DockerHub

## Required GitHub Secrets

To use this CI/CD pipeline, you need to set up the following secrets in your GitHub repository:

### DockerHub Secrets

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add the following repository secrets:

| Secret Name          | Description             | Example                 |
| -------------------- | ----------------------- | ----------------------- |
| `DOCKERHUB_USERNAME` | Your DockerHub username | `yourusername`          |
| `DOCKERHUB_TOKEN`    | DockerHub access token  | `dckr_pat_xxxxxxxxxxxx` |

### Creating DockerHub Access Token

1. Go to [DockerHub](https://hub.docker.com/)
2. Sign in to your account
3. Go to Account Settings → Security
4. Click "New Access Token"
5. Give it a descriptive name (e.g., "GitHub Actions CI")
6. Select appropriate permissions (Read & Write for pushing images)
7. Copy the generated token and add it as `DOCKERHUB_TOKEN` secret

## Pipeline Triggers

The pipeline runs on:

- **Push** to `main` or `dev` branches
- **Pull requests** to `main` branch

## Pipeline Jobs

### 1. Lint Job

- Sets up Python 3.10 environment
- Installs dependencies
- Runs code quality checks:
  - Black (code formatting)
  - isort (import sorting)
  - flake8 (linting)
  - mypy (type checking)

### 2. Test Job

- Depends on lint job completion
- Sets up Python 3.10 environment
- Installs dependencies
- Executes unit tests with pytest (no database required)
- **Enforces minimum coverage threshold of 1%** - pipeline fails if coverage is below this threshold
- Generates coverage reports (XML and HTML)
- Uploads coverage reports to Codecov

### 3. Build and Push Job

- Only runs on push to main/dev branches
- Depends on both lint and test jobs completion
- Builds Docker images for multiple architectures (linux/amd64, linux/arm64)
- Tags images with:
  - Branch name
  - Commit SHA
  - `latest` (for main branch only)
- Pushes images to DockerHub
- Uses Docker layer caching for faster builds

### 4. Security Scan Job

- Scans the built Docker image for vulnerabilities using Trivy
- Uploads results to GitHub Security tab

## Local Testing

To test the pipeline locally before pushing:

```bash
# Run code quality checks (lint job)
black --check .
isort --check-only .
flake8 .
mypy .

# Run unit tests (test job)
pytest tests/ -v --cov=app --cov-fail-under=1

# Check coverage threshold locally
coverage report --fail-under=1

# View HTML coverage report
open tests/htmlcov/index.html  # On macOS
start tests/htmlcov/index.html  # On Windows

# Build Docker image
docker build -t smartrent-ai:local .

# Test Docker image
docker run -p 8000:8000 smartrent-ai:local
```

## Troubleshooting

### Common Issues

1. **DockerHub authentication fails**:

   - Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set correctly
   - Check that the DockerHub token has appropriate permissions

2. **Tests fail**:

   - Ensure all dependencies are listed in `requirements.txt`
   - Check that unit tests don't require database connections
   - Mock any external dependencies in your tests
   - **Verify test coverage is above 1%** - add more tests if coverage is too low

3. **Coverage threshold not met**:

   - Review coverage report to identify untested code
   - Add unit tests for uncovered functions/classes
   - Check that tests are actually testing the `app` module code
   - View detailed HTML report at `tests/htmlcov/index.html` for line-by-line coverage

4. **Docker build fails**:

   - Verify the Dockerfile builds successfully locally
   - Check for any missing system dependencies

5. **Code quality checks fail**:
   - Run the checks locally and fix any issues before pushing
   - Ensure code follows the project's formatting standards

## Monitoring

- Check the Actions tab in your GitHub repository for pipeline status
- Review coverage reports uploaded to Codecov
- Monitor security scan results in the Security tab

## Next Steps

Consider adding:

- Automated deployment to staging environment
- Integration with monitoring tools
- Slack/email notifications for build failures
- Automated dependency updates with Dependabot
