# Contributing to Cross-Modal Neuromorphic Computing

Thank you for your interest in contributing! We welcome contributions from the community.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, PyTorch version)
- Error messages or logs

### Suggesting Enhancements

We welcome feature requests! Please create an issue with:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/cross-modal-neuromorphic.git
   cd cross-modal-neuromorphic
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding style (PEP 8)
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Provide clear description of changes
   - Reference related issues
   - Ensure all tests pass

## 📝 Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Type hints are encouraged

### Example
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 30
) -> Dict[str, float]:
    """
    Train a spiking neural network model.
    
    Args:
        model: SNN model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        num_epochs: Number of training epochs
        
    Returns:
        Dictionary containing training metrics
    """
    # Implementation
    pass
```

### Testing
- Write unit tests for new features
- Ensure tests pass before submitting PR
- Aim for >80% code coverage

### Documentation
- Update README.md for new features
- Add docstrings to functions
- Update relevant documentation in docs/

## 🔬 Research Contributions

### New Models
If you're adding a new memory mechanism or architecture:
1. Create new file in `models/`
2. Follow existing model structure
3. Add configuration in `experiments/`
4. Add tests in `tests/`
5. Document in README and relevant docs

### New Datasets
If you're adding support for a new neuromorphic dataset:
1. Create loader in `data/`
2. Follow existing loader structure
3. Add download script
4. Update documentation

### New Analysis
If you're adding new analysis methods:
1. Create script in `analysis/`
2. Add visualization utilities
3. Document usage
4. Add example notebook

## 📊 Reproducing Results

Before submitting changes that affect model performance:
1. Run full ablation study
2. Compare results with baseline
3. Document any performance changes
4. Include comparison in PR description

## 🐛 Debugging Tips

### Common Issues
- **CUDA out of memory**: Reduce batch size in config
- **Import errors**: Check `requirements.txt` is up to date
- **Dataset not found**: Run download scripts
- **Model convergence**: Check learning rate and optimizer settings

### Getting Help
- Check existing issues
- Review documentation
- Ask questions in issue tracker
- Contact maintainers

## 📧 Contact

- **Maintainer**: Effiong Blessing (blessing.effiong@slu.edu)
- **Issues**: Use GitHub issue tracker
- **Discussions**: Use GitHub Discussions

## 🎯 Priorities

Current priorities for contributions:
1. Support for additional neuromorphic datasets
2. Hardware validation on Loihi/BrainScaleS
3. Transfer learning experiments
4. Additional memory mechanisms
5. Improved documentation

## ✅ Checklist for Contributors

Before submitting your PR, ensure:
- [ ] Code follows PEP 8 style guidelines
- [ ] All tests pass (`pytest tests/`)
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main
- [ ] PR description is detailed

Thank you for contributing! 🙏
