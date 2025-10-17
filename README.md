# 🖼️ SqueezeNet Image Classification Application

A comprehensive, production-ready application for image classification using SqueezeNet deep learning models. This application provides both command-line and web-based interfaces for classifying images into 1000 ImageNet categories with high efficiency and accuracy.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![ONNX](https://img.shields.io/badge/ONNX-1.19-green)](https://onnx.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Web Interface](#web-interface)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Examples](#examples)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## 🎯 Overview

**SqueezeNet** is a highly efficient convolutional neural network architecture that achieves AlexNet-level accuracy with **50x fewer parameters** and a model size of **less than 0.5MB**. This makes it ideal for:

- 📱 Mobile and embedded applications
- 🌐 Edge computing devices
- ⚡ Real-time image classification
- 💾 Platforms with strict size and memory constraints

### Use Case

SqueezeNet models perform **image classification** - they take images as input and classify the major object in the image into one of **1000 pre-defined ImageNet classes**. The models are trained on the ImageNet dataset which contains images from categories including:

- 🐕 Animals (dogs, cats, birds, fish, etc.)
- 🚗 Vehicles (cars, trucks, airplanes, boats)
- 🪑 Objects (furniture, electronics, instruments)
- 🍎 Food items (fruits, vegetables, dishes)
- 🌳 Nature elements (trees, flowers, landscapes)

## ✨ Features

### Core Features
- ✅ **Pre-trained SqueezeNet Models**: Support for both SqueezeNet 1.0 and 1.1
- ✅ **ONNX Runtime**: Fast inference using optimized ONNX Runtime
- ✅ **Multiple Interfaces**: CLI, Web UI, and Python API
- ✅ **Batch Processing**: Efficiently process multiple images
- ✅ **Visual Output**: Generate annotated images with predictions
- ✅ **Detailed Results**: Get top-K predictions with confidence scores

### User Interfaces
- 🖥️ **Command Line Interface**: Powerful CLI for batch processing and scripting
- 🌐 **Web Interface**: Beautiful Gradio-based web UI for interactive use
- 🐍 **Python API**: Easy-to-use API for integration into your projects

### Additional Features
- 📊 Performance measurement and benchmarking
- 🎨 Image preprocessing and augmentation utilities
- 📈 Model comparison tools
- 🔍 Detailed model information and diagnostics
- 📝 Comprehensive examples and documentation

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Internet connection (for model download)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ImageClassification_SqueezeNet.git
cd ImageClassification_SqueezeNet
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `onnxruntime` - Fast inference engine
- `numpy` - Numerical computing
- `Pillow` - Image processing
- `opencv-python` - Computer vision utilities
- `gradio` - Web interface
- `requests` - HTTP library for downloads
- `tqdm` - Progress bars

### Step 3: Download Models

```bash
python setup.py
```

This will download:
- SqueezeNet 1.1 ONNX model (~5 MB)
- ImageNet class labels (~100 KB)

## ⚡ Quick Start

### 1. Web Interface (Easiest)

Launch the interactive web application:

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:7860`

### 2. Command Line

Classify a single image:

```bash
python classify.py --image path/to/your/image.jpg
```

### 3. Python Script

```python
from src.squeezenet_classifier import SqueezeNetClassifier

# Initialize classifier
classifier = SqueezeNetClassifier(
    model_path='models/squeezenet1.1-7.onnx',
    labels_path='models/imagenet_labels.json'
)

# Classify an image
predictions = classifier.predict('image.jpg', top_k=5)

# Display results
for class_name, probability in predictions:
    print(f"{class_name}: {probability*100:.2f}%")
```

## 📖 Usage

### Web Interface

The web interface provides the most user-friendly way to classify images.

**Starting the server:**

```bash
# Basic usage
python app.py

# Custom host and port
python app.py --host 0.0.0.0 --port 8080

# Create public shareable link
python app.py --share
```

**Features:**
- 📤 Upload images via drag-and-drop or file picker
- 📷 Use webcam for real-time classification
- 📋 Copy-paste images from clipboard
- 📊 View top-10 predictions with confidence bars
- 💾 Download classified results

### Command Line Interface

The CLI provides powerful options for automation and batch processing.

**Basic Usage:**

```bash
# Classify a single image
python classify.py --image cat.jpg

# Get top 10 predictions
python classify.py --image dog.jpg --top-k 10

# Save annotated output
python classify.py --image bird.jpg --save

# Specify custom output path
python classify.py --image test.jpg --output results/annotated.jpg
```

**Batch Processing:**

```bash
# Process all images in a directory
python classify.py --batch-dir images/

# Use custom model
python classify.py --batch-dir photos/ --model models/squeezenet1.0-12.onnx
```

**Model Information:**

```bash
# Display model details
python classify.py --model-info
```

**CLI Options:**

```
Input Options:
  --image, -i PATH          Path to input image
  --batch-dir, -b PATH      Directory for batch processing
  --model-info              Show model information

Model Options:
  --model, -m PATH          Path to ONNX model (default: models/squeezenet1.1-7.onnx)
  --labels, -l PATH         Path to labels JSON (default: models/imagenet_labels.json)

Output Options:
  --output, -o PATH         Save annotated image to path
  --save, -s                Save with default name
  --top-k, -k NUM           Number of predictions (default: 5)
  --info                    Show image information
```

### Python API

Integrate SqueezeNet into your Python applications.

**Basic Classification:**

```python
from src.squeezenet_classifier import SqueezeNetClassifier

# Initialize
classifier = SqueezeNetClassifier(
    model_path='models/squeezenet1.1-7.onnx',
    labels_path='models/imagenet_labels.json'
)

# Single image
predictions = classifier.predict('image.jpg', top_k=5)
for class_name, prob in predictions:
    print(f"{class_name}: {prob:.3f}")
```

**Batch Processing:**

```python
# Multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = classifier.predict_batch(image_paths, top_k=3)

for path, preds in zip(image_paths, results):
    print(f"\n{path}:")
    for class_name, prob in preds:
        print(f"  {class_name}: {prob:.3f}")
```

**Custom Preprocessing:**

```python
# Preprocess manually
preprocessed = classifier.preprocess_image('image.jpg')
print(f"Shape: {preprocessed.shape}")  # (1, 3, 224, 224)

# Get model information
info = classifier.get_model_info()
print(info)
```

**Using Image Utilities:**

```python
from utils.image_utils import draw_predictions, get_image_info

# Classify and annotate
predictions = classifier.predict('image.jpg')
annotated_path = draw_predictions('image.jpg', predictions, 'output.jpg')

# Get image info
info = get_image_info('image.jpg')
print(f"Size: {info['width']}x{info['height']}")
```

## 📁 Project Structure

```
ImageClassification_SqueezeNet/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── setup.py                   # Model download script
├── classify.py                # CLI interface
├── app.py                     # Web interface
│
├── src/                       # Core source code
│   ├── __init__.py
│   └── squeezenet_classifier.py   # Main classifier class
│
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── model_downloader.py   # Model download utilities
│   └── image_utils.py         # Image processing utilities
│
├── examples/                  # Example scripts
│   ├── __init__.py
│   ├── basic_usage.py        # Basic usage example
│   ├── batch_processing.py   # Batch processing example
│   └── advanced_usage.py     # Advanced features example
│
├── models/                    # Model files (downloaded)
│   ├── README.md
│   ├── squeezenet1.1-7.onnx  # SqueezeNet 1.1 model
│   └── imagenet_labels.json  # Class labels
│
├── images/                    # Sample images directory
│   └── README.md
│
└── outputs/                   # Output directory (created automatically)
```

## 🤖 Model Information

### SqueezeNet 1.1 (Default)

- **Size**: ~5 MB
- **Parameters**: ~1.2 million
- **Input**: 224×224×3 RGB images
- **Output**: 1000 class probabilities
- **Speed**: ~2.3x faster than SqueezeNet 1.0
- **Accuracy**: 
  - Top-1: ~57.4%
  - Top-5: ~80.2%

### SqueezeNet 1.0 (Alternative)

- **Size**: ~5 MB
- **Parameters**: ~1.2 million
- **Input**: 224×224×3 RGB images
- **Output**: 1000 class probabilities
- **Accuracy**:
  - Top-1: ~57.5%
  - Top-5: ~80.3%

### Architecture Highlights

SqueezeNet uses innovative "Fire modules" consisting of:

1. **Squeeze Layer**: 1×1 convolutions to reduce channels
2. **Expand Layer**: Mix of 1×1 and 3×3 convolutions

This design achieves:
- 50x fewer parameters than AlexNet
- Similar accuracy to AlexNet
- Model size < 0.5 MB (with compression)

## 📚 Examples

The `examples/` directory contains comprehensive example scripts:

### Basic Usage

```bash
python examples/basic_usage.py
```

Demonstrates:
- Loading the classifier
- Classifying a single image
- Displaying results

### Batch Processing

```bash
python examples/batch_processing.py
```

Demonstrates:
- Processing multiple images
- Saving annotated outputs
- Generating summary reports

### Advanced Usage

```bash
python examples/advanced_usage.py
```

Demonstrates:
- Performance measurement
- Model comparison (1.0 vs 1.1)
- Preprocessing analysis
- Custom inference workflows

## ⚡ Performance

### Inference Speed

On a typical modern CPU:
- **Single image**: ~50-100ms
- **Batch (10 images)**: ~500-800ms
- **Throughput**: ~10-20 images/second

On GPU (CUDA):
- **Single image**: ~5-10ms
- **Throughput**: ~100-200 images/second

### Memory Usage

- **Model size**: ~5 MB
- **Runtime memory**: ~50-100 MB
- **Peak memory**: ~200-300 MB (including image processing)

### Accuracy

Trained on ImageNet 2012 validation set:
- **Top-1 Accuracy**: 57.4%
- **Top-5 Accuracy**: 80.2%

## 🔧 Troubleshooting

### Model not found error

```
Error: Model file not found
```

**Solution**: Run the setup script to download models:
```bash
python setup.py
```

### Import errors

```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Image loading errors

```
Error: Invalid image file
```

**Solution**: 
- Ensure image format is supported (JPEG, PNG, BMP, GIF, WebP)
- Check file is not corrupted
- Verify file path is correct

### Performance issues

If inference is slow:
- Use SqueezeNet 1.1 (faster than 1.0)
- Enable GPU support: `pip install onnxruntime-gpu`
- Reduce image resolution before classification
- Use batch processing for multiple images

### Out of memory errors

**Solution**:
- Process images in smaller batches
- Reduce image resolution
- Close other applications
- Use 64-bit Python

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs**: Open an issue describing the bug
2. **Suggest features**: Open an issue with your feature idea
3. **Submit pull requests**: Fork, create a branch, make changes, submit PR
4. **Improve documentation**: Help make the docs clearer
5. **Add examples**: Share your use cases and examples

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ImageClassification_SqueezeNet.git
cd ImageClassification_SqueezeNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8

# Run tests
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📖 References

### Research Papers

- **SqueezeNet**: Iandola et al., "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" (2016)
  - Paper: https://arxiv.org/abs/1602.07360
  
### Resources

- **ONNX Model Zoo**: https://github.com/onnx/models
- **ImageNet Dataset**: http://www.image-net.org/
- **Original Implementation**: https://github.com/forresti/SqueezeNet
- **ONNX Runtime**: https://onnxruntime.ai/

### Related Projects

- PyTorch implementation: https://pytorch.org/vision/stable/models.html
- TensorFlow implementation: https://github.com/DT42/squeezenet_demo
- Caffe models: https://github.com/DeepScale/SqueezeNet

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SqueezeNet Authors**: Forrest Iandola, Song Han, Matthew Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
- **ONNX Model Zoo**: For providing pre-trained models
- **ImageNet**: For the comprehensive dataset
- **Contributors**: Thanks to all who have contributed to this project

## 📞 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Email: your.email@example.com
- Twitter: @yourusername

---

**Made with ❤️ for the computer vision community**

*Star ⭐ this repository if you find it helpful!*
