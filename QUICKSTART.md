# 🚀 Quick Start Guide

Get started with SqueezeNet Image Classification in 5 minutes!

## 1. Setup (2 minutes)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Models

```bash
python setup.py
```

Press 'Y' when prompted. This will download ~5 MB of models.

## 2. Try It Out! (3 minutes)

### Option A: Web Interface (Recommended for Beginners)

```bash
python app.py
```

1. Open browser to http://127.0.0.1:7860
2. Drag and drop an image
3. See instant results!

### Option B: Command Line

```bash
python classify.py --image path/to/your/image.jpg
```

### Option C: Python Code

Create `test.py`:

```python
from src.squeezenet_classifier import SqueezeNetClassifier

classifier = SqueezeNetClassifier(
    'models/squeezenet1.1-7.onnx',
    'models/imagenet_labels.json'
)

predictions = classifier.predict('your_image.jpg', top_k=5)

for class_name, prob in predictions:
    print(f"{class_name}: {prob*100:.1f}%")
```

Run it:
```bash
python test.py
```

## 3. What's Next?

### Try More Features

```bash
# Batch process multiple images
python classify.py --batch-dir images/

# Get top 10 predictions
python classify.py --image test.jpg --top-k 10

# Save annotated output
python classify.py --image test.jpg --save

# Check model info
python classify.py --model-info
```

### Explore Examples

```bash
# Basic usage
python examples/basic_usage.py

# Batch processing
python examples/batch_processing.py

# Advanced features
python examples/advanced_usage.py
```

### Customize

```bash
# Use custom model
python classify.py --image test.jpg --model models/squeezenet1.0-12.onnx

# Custom host/port for web app
python app.py --host 0.0.0.0 --port 8080

# Create public link
python app.py --share
```

## 4. Common Issues

### "Model not found"
```bash
python setup.py
```

### "Module not found"
```bash
pip install -r requirements.txt
```

### Slow performance
- Use SqueezeNet 1.1 (default)
- Install GPU support: `pip install onnxruntime-gpu`

## 5. Best Practices

### Good Images for Testing
- Clear, well-lit images
- Single prominent object
- Common objects (animals, vehicles, everyday items)
- Avoid extreme angles or heavy filters

### ImageNet Categories
The model recognizes 1000 categories including:
- Animals: dog, cat, bird, elephant, etc.
- Vehicles: car, airplane, bicycle, etc.
- Objects: chair, laptop, guitar, etc.
- Food: banana, pizza, strawberry, etc.

### Tips for Best Results
- Use high-quality images
- Center the main object
- Ensure good lighting
- Avoid cluttered backgrounds
- Use common ImageNet objects

## 6. Learn More

- **Full Documentation**: See [README.md](README.md)
- **Examples**: Check the `examples/` directory
- **API Reference**: See docstrings in `src/squeezenet_classifier.py`
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Need Help?

- Check the [Troubleshooting](README.md#troubleshooting) section
- Open an issue on GitHub
- Read the full documentation

Happy classifying! 🎉

