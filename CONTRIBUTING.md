# Contributing to M2M Vector Search

Thank you for your interest in contributing to M2M Vector Search!

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/m2m-vector-search.git
   cd m2m-vector-search
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e .[dev]
   pip install pre-commit
   pre-commit install
   ```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatter (line length: 100)
- **isort**: Import sorter
- **flake8**: Linter
- **mypy**: Type checker
- **pytest**: Testing framework

### Running checks

```bash
# Format code
black src/ test_*.py

# Sort imports
isort src/ test_*.py

# Check linting
flake8 src/ test_*.py

# Type checking
mypy src/

# Run tests
pytest test_m2m_advanced.py -v

# Run all pre-commit hooks
pre-commit run --all-files
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Ensure all tests pass
   - Update documentation if needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Test updates
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI passes

## Code Guidelines

### Python Style

- Follow PEP 8
- Use type hints for all public APIs
- Write docstrings for all public functions/classes
- Maximum line length: 100 characters
- Use f-strings for string formatting

### Testing

- Write tests for all new functionality
- Maintain or improve test coverage
- Use descriptive test names
- Test edge cases and error conditions

### Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Keep CHANGELOG.md updated

## Questions or Issues?

- Open an issue on GitHub
- Check existing documentation
- Review closed issues/PRs

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing! 🎉
