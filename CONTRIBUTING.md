# Contributing to AKM

Thank you for your interest in contributing to the Adaptive Knowledge Mesh (AKM) framework! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/akm.git
cd akm
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/unit/ -v

# Run linting
ruff check akm/
black --check akm/
mypy akm/
```

## Development Workflow

### 1. Create a Branch

```bash
# Create a feature branch from main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Changes

- Write clean, readable code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format code
black akm/ tests/
ruff check akm/ --fix

# Type checking
mypy akm/

# Run tests
pytest tests/ -v --cov=akm
```

### 4. Commit Changes

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Format: <type>(<scope>): <description>

git commit -m "feat(links): add semantic similarity pattern detection"
git commit -m "fix(decay): correct exponential decay formula"
git commit -m "docs(readme): add installation instructions"
git commit -m "test(mapper): add tests for entity type inference"
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows the project style guidelines
- [ ] Tests pass locally (`pytest tests/`)
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages follow conventional commits
- [ ] PR description explains the changes

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested these changes.

## Related Issues
Fixes #123
```

## Code Style

### Python Style

We use the following tools:

- **Black**: Code formatting (line length: 88)
- **Ruff**: Linting
- **mypy**: Type checking

### Guidelines

```python
# Use type hints
def process_entity(entity: Entity, transformer: DomainTransformer) -> Entity:
    """Process an entity through the domain transformer.

    Args:
        entity: The entity to process.
        transformer: The domain transformer to apply.

    Returns:
        The transformed entity.

    Raises:
        ValidationError: If entity validation fails.
    """
    return transformer.map_entity(entity)


# Use dataclasses or Pydantic for data structures
from pydantic import BaseModel

class Config(BaseModel):
    name: str
    value: int = 0


# Prefer explicit imports
from akm.core.models import Entity, Relationship
# Not: from akm.core.models import *
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_decay(weight: float, decay_rate: float, hours: float) -> float:
    """Calculate exponential decay for a link weight.

    Uses the formula: weight * exp(-decay_rate * hours)

    Args:
        weight: Current weight value (0.0 to 1.0).
        decay_rate: Decay rate per hour.
        hours: Number of hours elapsed.

    Returns:
        New weight after decay applied.

    Example:
        >>> calculate_decay(0.5, 0.01, 24)
        0.3935
    """
    return weight * math.exp(-decay_rate * hours)
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests (no external dependencies)
│   ├── core/
│   ├── links/
│   └── domain/
├── integration/    # Integration tests (may need Docker)
│   ├── test_neo4j.py
│   └── test_chromadb.py
└── e2e/            # End-to-end tests
    └── test_full_workflow.py
```

### Writing Tests

```python
import pytest
from akm.core.models import Entity, LinkWeight


class TestLinkWeight:
    """Tests for LinkWeight model."""

    def test_decay_reduces_weight(self):
        """Decay should reduce weight over time."""
        weight = LinkWeight(value=1.0, decay_rate=0.01)
        new_value = weight.apply_decay(24)  # 24 hours

        assert new_value < 1.0
        assert new_value > 0.0

    def test_validation_increases_weight(self):
        """Positive validation should increase weight."""
        weight = LinkWeight(value=0.5)
        weight.apply_validation(is_positive=True)

        assert weight.value == 0.6
        assert weight.positive_validations == 1

    @pytest.mark.parametrize("initial,expected", [
        (0.9, True),   # Above promotion threshold
        (0.7, False),  # Below promotion threshold
    ])
    def test_should_promote(self, initial, expected):
        """Test promotion threshold logic."""
        weight = LinkWeight(value=initial, promotion_threshold=0.8)
        assert weight.should_promote() == expected
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/links/test_decay.py

# Run with coverage
pytest --cov=akm --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run integration tests (requires Docker)
docker-compose up -d
pytest tests/integration/
```

## Project Structure

```
akm/
├── core/           # Data models, interfaces, config
│   ├── models.py       # Entity, Relationship, Link, etc.
│   ├── interfaces.py   # Abstract protocols
│   ├── config.py       # Pydantic configuration
│   └── exceptions.py   # Custom exceptions
├── graph/          # Graph backends
│   ├── neo4j/          # Neo4j implementation
│   └── memory/         # In-memory implementation
├── vector/         # Vector backends
│   ├── chromadb/       # ChromaDB implementation
│   └── embeddings/     # Embedding models
├── links/          # Adaptive link management
│   ├── manager.py      # LinkManager
│   ├── decay.py        # Decay logic
│   └── validation.py   # User validation
├── domain/         # Domain transformation
│   ├── transformer.py  # BaseDomainTransformer
│   └── schema/         # Schema loading and mapping
├── query/          # Query engine
├── ingestion/      # Data ingestion
├── gnn/            # Graph neural networks
├── api/            # Public API
└── cli/            # Command-line interface
```

## Adding New Features

### Adding a New Backend

1. Create interface in `akm/core/interfaces.py`
2. Implement in appropriate directory (e.g., `akm/graph/newdb/`)
3. Add factory method in base module
4. Add configuration options
5. Write tests
6. Update documentation

### Adding a New Domain Adapter

1. Create schema YAML in `akm/adapters/your_domain/schema.yaml`
2. Optionally extend `BaseDomainTransformer`
3. Add tests in `tests/unit/domain/`
4. Document in `docs/DOMAIN_TRANSFORMER.md`

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a release PR
4. After merge, tag the release
5. GitHub Actions publishes to PyPI

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@akm-framework.org

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to AKM!
