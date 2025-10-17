# Models Directory

This directory stores the ONNX models and label files for SqueezeNet image classification.

## Automatic Setup

Run the setup script to download models automatically:

```bash
python setup.py
```

This will download:
- **SqueezeNet 1.1 ONNX model** (~5 MB)
- **ImageNet class labels** (~100 KB)

## Manual Download

If you prefer to download manually:

### SqueezeNet 1.1 (Recommended)
```bash
# Download from ONNX Model Zoo
wget https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
```

### SqueezeNet 1.0 (Alternative)
```bash
wget https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx
```

### ImageNet Labels
```bash
wget https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json -O imagenet_labels.json
```

## Model Details

### SqueezeNet 1.1
- **Size**: ~5 MB
- **Parameters**: ~1.2M
- **Input**: 224×224×3 RGB images
- **Output**: 1000 class probabilities
- **Performance**: ~2.3x faster than SqueezeNet 1.0
- **Accuracy**: Top-1: ~57.4%, Top-5: ~80.2%

### SqueezeNet 1.0
- **Size**: ~5 MB
- **Parameters**: ~1.2M
- **Input**: 224×224×3 RGB images
- **Output**: 1000 class probabilities
- **Accuracy**: Top-1: ~57.5%, Top-5: ~80.3%

## ImageNet Classes

The model is trained on ImageNet dataset with 1000 object categories including:
- Animals (dog, cat, bird, fish, etc.)
- Vehicles (car, truck, airplane, etc.)
- Objects (furniture, electronics, tools, etc.)
- Food items
- Nature elements

For the complete list of 1000 classes, see: http://www.image-net.org/

## Model Architecture

SqueezeNet uses a unique architecture called "Fire modules" that consists of:
1. **Squeeze layer**: 1×1 convolutions to reduce channels
2. **Expand layer**: Mix of 1×1 and 3×3 convolutions

This design achieves AlexNet-level accuracy with 50x fewer parameters.

## Using Different Models

To use a specific model with the CLI:

```bash
python classify.py --image test.jpg --model models/squeezenet1.0-12.onnx
```

To use a specific model with the web app:

```bash
python app.py --model models/squeezenet1.1-7.onnx
```

## References

- **Paper**: [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters](https://arxiv.org/abs/1602.07360)
- **ONNX Model Zoo**: https://github.com/onnx/models
- **Original Implementation**: https://github.com/forresti/SqueezeNet

