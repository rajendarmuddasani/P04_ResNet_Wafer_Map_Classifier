# Contributing to ResNet Wafer Map Classifier

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

- **Bug Reports**: Report bugs via GitHub Issues
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests for bug fixes or features
- **Documentation**: Improve documentation, examples, or tutorials
- **Testing**: Add test coverage or report test failures

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/P04_ResNet_Wafer_Map_Classifier.git
   cd P04_ResNet_Wafer_Map_Classifier
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 📝 Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Write docstrings for all public functions/classes
- Keep line length under 100 characters
- Use meaningful variable and function names

## 🧪 Testing

Before submitting a PR, ensure all tests pass:

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Check coverage
pytest --cov=src tests/
```

## 📤 Submitting Changes

1. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Use conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Wait for review

## 🔍 Code Review Process

- All PRs require at least one approval
- Address reviewer comments promptly
- Keep PR scope focused and manageable
- Update documentation if changing APIs

## 📋 Pull Request Checklist

- [ ] Code follows PEP 8 style guide
- [ ] Added/updated tests for changes
- [ ] All tests pass locally
- [ ] Updated documentation
- [ ] Added type hints
- [ ] No sensitive information (API keys, passwords)
- [ ] Commit messages follow conventional format

## 🐛 Reporting Bugs

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Logs**: Relevant error messages or logs

## 💡 Feature Requests

When requesting features, include:

- **Problem**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions you considered
- **Use Case**: Real-world example of usage

## 📞 Questions?

- Open a GitHub Discussion for general questions
- Check existing issues before creating new ones
- Be respectful and constructive in all interactions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making semiconductor manufacturing more efficient! 🙏
