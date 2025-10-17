# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-17

### Added
- Initial release of SqueezeNet Image Classification Application
- Core SqueezeNet classifier module with ONNX Runtime support
- Command-line interface for image classification
- Web-based interface using Gradio
- Automatic model downloader from ONNX Model Zoo
- Support for both SqueezeNet 1.0 and 1.1 models
- Batch processing capabilities
- Image preprocessing and annotation utilities
- Comprehensive documentation and examples
- Example scripts:
  - Basic usage example
  - Batch processing example
  - Advanced usage with performance benchmarks
- Test installation script
- Quick start guide
- Contributing guidelines
- MIT License

### Features
- Single image classification
- Batch directory processing
- Top-K predictions with confidence scores
- Annotated output images with predictions overlay
- Model information and diagnostics
- Performance measurement tools
- ImageNet 1000-class classification
- Web UI with drag-and-drop support
- Webcam integration for real-time classification
- CLI with extensive options
- Python API for easy integration

### Documentation
- Comprehensive README with usage examples
- API documentation in code
- Separate documentation for models and images directories
- Quick start guide for beginners
- Contributing guidelines
- Troubleshooting section

### Dependencies
- onnxruntime 1.19.2
- numpy 1.26.4
- Pillow 10.4.0
- opencv-python 4.10.0.84
- gradio 4.44.0
- requests 2.32.3
- tqdm 4.66.5

## Upcoming Features (Planned)

### Version 1.1.0
- [ ] GPU acceleration support
- [ ] Docker container support
- [ ] REST API endpoint
- [ ] More output formats (JSON, CSV)
- [ ] Custom model support
- [ ] Image augmentation options

### Version 1.2.0
- [ ] Video classification support
- [ ] Real-time webcam classification
- [ ] Batch export functionality
- [ ] Performance optimization
- [ ] Mobile app integration examples

### Future
- [ ] Support for other ONNX models
- [ ] Transfer learning capabilities
- [ ] Model fine-tuning tools
- [ ] Cloud deployment guides
- [ ] Kubernetes deployment
- [ ] CI/CD pipeline
- [ ] Automated testing suite

## Notes

For detailed changes in each release, see the [GitHub releases page](https://github.com/yourusername/ImageClassification_SqueezeNet/releases).

