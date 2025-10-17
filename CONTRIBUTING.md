# Contributing to SqueezeNet Image Classification

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, screenshots, etc.)
- **Describe the behavior you observed and what you expected**
- **Include your environment details** (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **Provide examples of how it would work**

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Make your changes** following the coding standards
3. **Add or update tests** if applicable
4. **Update documentation** if needed
5. **Ensure tests pass**
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ImageClassification_SqueezeNet.git
cd ImageClassification_SqueezeNet

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Code Formatting

We use `black` for code formatting:

```bash
black src/ utils/ examples/
```

### Linting

Run `flake8` to check for issues:

```bash
flake8 src/ utils/ examples/
```

### Type Hints

Add type hints to function signatures:

```python
def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """Classify an image."""
    pass
```

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use descriptive test names

Example:

```python
def test_classifier_initialization():
    """Test that classifier initializes correctly."""
    classifier = SqueezeNetClassifier(model_path, labels_path)
    assert classifier is not None
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### README Updates

If your changes affect usage or features:
- Update the main README.md
- Add examples if introducing new features
- Update the Quick Start section if needed

## Commit Messages

Write clear and meaningful commit messages:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

Examples:
```
Add batch processing support for multiple images

- Implement batch_predict method
- Add CLI option for batch processing
- Update documentation

Fixes #123
```

## Project Structure

When adding new files, follow the existing structure:

```
src/           # Core application code
utils/         # Utility functions and helpers
examples/      # Example scripts
tests/         # Test files
models/        # Model files (not committed)
images/        # Sample images
```

## Review Process

1. **Automated checks** will run on your PR
2. **Code review** by maintainers
3. **Discussion** if changes are needed
4. **Merge** when approved

## Questions?

Feel free to ask questions by:
- Opening an issue
- Commenting on existing issues
- Reaching out to maintainers

## Code of Conduct

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Enforcement

Unacceptable behavior may be reported to project maintainers. All complaints will be reviewed and investigated.

## Recognition

Contributors will be recognized in:
- The README.md file
- Release notes
- Project documentation

Thank you for contributing! 🚀

