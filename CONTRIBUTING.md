# Contributing to High-Frequency Statistical Arbitrage Strategy Development

Thank you for your interest in contributing to this project! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** for your changes
4. **Make your changes** following the guidelines below
5. **Test your changes** thoroughly
6. **Submit a pull request** with a clear description

## 📋 Development Setup

### Prerequisites
- C++17 compiler (GCC 9+ or Clang 12+)
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- CMake 3.16+

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/high-frequency-arbitrage.git
cd high-frequency-arbitrage

# Setup Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r python/requirements.txt

# Build C++ components
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..
```

## 🔧 Development Guidelines

### Code Style

#### C++ Code
- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use meaningful variable and function names
- Add comprehensive comments for complex algorithms
- Include header guards in all header files
- Use `const` where appropriate

#### Python Code
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Add docstrings for all public functions and classes
- Keep functions focused and under 50 lines when possible

### Testing

#### C++ Testing
```bash
cd cpp/build
make test
```

#### Python Testing
```bash
# Run all tests
python -m pytest python/tests/

# Run specific test file
python -m pytest python/tests/test_alpha_generation.py

# Run with coverage
python -m pytest python/tests/ --cov=python/src --cov-report=html
```

### Documentation

- Update the README.md if you add new features
- Add docstrings to all new Python functions and classes
- Update relevant documentation in the `docs/` directory
- Include code examples for new features

## 🎯 Areas for Contribution

### High Priority
- **Performance Optimization**: Improve C++ pipeline latency
- **New Alpha Models**: Implement additional ML models
- **Risk Management**: Enhance position sizing and risk controls
- **Testing**: Add comprehensive unit and integration tests

### Medium Priority
- **Documentation**: Improve code documentation and tutorials
- **Configuration**: Add more configuration options
- **Monitoring**: Enhance logging and monitoring capabilities
- **GPU Optimization**: Improve CUDA kernel performance

### Low Priority
- **UI/UX**: Create web-based dashboard
- **Alternative Data**: Integrate news sentiment, social media data
- **Multi-Asset**: Extend to forex, crypto, commodities
- **Research**: Implement cutting-edge academic papers

## 📝 Pull Request Process

1. **Create a descriptive title** for your PR
2. **Provide a detailed description** of your changes
3. **Include tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** before submitting
6. **Request review** from maintainers

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Test addition

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment details**: OS, Python version, C++ compiler
2. **Steps to reproduce**: Clear, step-by-step instructions
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full error traceback
6. **Screenshots**: If applicable

## 💡 Feature Requests

When requesting features, please include:

1. **Problem description**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Use cases**: Who would benefit from this?
4. **Implementation ideas**: Any technical suggestions?

## 📊 Performance Contributions

For performance improvements:

1. **Benchmark current performance** using provided tools
2. **Implement optimization** with clear rationale
3. **Measure improvement** with same benchmarks
4. **Document trade-offs** (memory vs speed, etc.)

## 🔒 Security

- **Never commit API keys** or sensitive credentials
- **Use environment variables** for configuration
- **Validate all inputs** to prevent injection attacks
- **Follow secure coding practices** for financial applications

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Wiki**: For detailed documentation and tutorials

## 🏆 Recognition

Contributors will be recognized in:
- **README.md** contributors section
- **Release notes** for significant contributions
- **GitHub contributors** page

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the High-Frequency Statistical Arbitrage Strategy Development project! 🚀 