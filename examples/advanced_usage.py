#!/usr/bin/env python3
"""
Advanced Usage Example

This example demonstrates advanced features like custom preprocessing,
model information retrieval, and performance measurement.
"""

import sys
import os
import time
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.squeezenet_classifier import SqueezeNetClassifier
from utils.image_utils import get_image_info


def measure_inference_time(classifier, image_path, num_runs=10):
    """Measure average inference time."""
    print(f"\nMeasuring inference time ({num_runs} runs)...")
    
    times = []
    for i in range(num_runs):
        start_time = time.time()
        predictions = classifier.predict(image_path, top_k=5)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Average inference time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    print(f"Throughput: {1/avg_time:.2f} images/second")
    
    return avg_time


def compare_models(image_path):
    """Compare SqueezeNet 1.0 vs 1.1 performance."""
    models = {
        'SqueezeNet 1.0': 'models/squeezenet1.0-12.onnx',
        'SqueezeNet 1.1': 'models/squeezenet1.1-7.onnx'
    }
    
    labels_path = 'models/imagenet_labels.json'
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\n{model_name}: Not available")
            continue
        
        print(f"\n{model_name}:")
        print("-" * 70)
        
        # Load model
        classifier = SqueezeNetClassifier(model_path, labels_path)
        
        # Model info
        info = classifier.get_model_info()
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {model_size:.2f} MB")
        print(f"Input shape: {info['input_shape']}")
        
        # Measure performance
        avg_time = measure_inference_time(classifier, image_path, num_runs=5)
        
        # Get predictions
        predictions = classifier.predict(image_path, top_k=3)
        print("\nTop 3 predictions:")
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"  {i}. {class_name}: {prob*100:.2f}%")


def analyze_image_preprocessing(image_path):
    """Demonstrate the preprocessing steps."""
    print("\n" + "="*70)
    print("IMAGE PREPROCESSING ANALYSIS")
    print("="*70)
    
    # Original image info
    info = get_image_info(image_path)
    print(f"\nOriginal Image:")
    print(f"  Path: {image_path}")
    print(f"  Format: {info['format']}")
    print(f"  Size: {info['width']}x{info['height']}")
    print(f"  Mode: {info['mode']}")
    print(f"  File size: {info['file_size_mb']:.2f} MB")
    
    # Load and preprocess
    model_path = 'models/squeezenet1.1-7.onnx'
    labels_path = 'models/imagenet_labels.json'
    
    classifier = SqueezeNetClassifier(model_path, labels_path)
    
    # Preprocess
    preprocessed = classifier.preprocess_image(image_path)
    
    print(f"\nPreprocessed Array:")
    print(f"  Shape: {preprocessed.shape}")
    print(f"  Data type: {preprocessed.dtype}")
    print(f"  Min value: {preprocessed.min():.3f}")
    print(f"  Max value: {preprocessed.max():.3f}")
    print(f"  Mean value: {preprocessed.mean():.3f}")
    print(f"  Std value: {preprocessed.std():.3f}")
    
    print(f"\nPreprocessing Steps:")
    print(f"  1. Resize to 224x224")
    print(f"  2. Convert to RGB")
    print(f"  3. Normalize to [0, 1]")
    print(f"  4. Apply ImageNet mean/std normalization")
    print(f"  5. Convert from HWC to CHW format")
    print(f"  6. Add batch dimension")


def main():
    """Main function for advanced examples."""
    
    # Check model
    model_path = 'models/squeezenet1.1-7.onnx'
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run 'python setup.py' first.")
        return 1
    
    # Example image
    image_path = 'images/cat.jpg'
    if not os.path.exists(image_path):
        print(f"\nWarning: Example image not found: {image_path}")
        print("Please provide your own image path.")
        return 1
    
    print("\n" + "="*70)
    print("SQUEEZENET ADVANCED USAGE EXAMPLES")
    print("="*70)
    
    # 1. Analyze preprocessing
    analyze_image_preprocessing(image_path)
    
    # 2. Measure performance
    labels_path = 'models/imagenet_labels.json'
    classifier = SqueezeNetClassifier(model_path, labels_path)
    measure_inference_time(classifier, image_path, num_runs=10)
    
    # 3. Compare models (if available)
    if os.path.exists('models/squeezenet1.0-12.onnx'):
        compare_models(image_path)
    else:
        print("\n💡 Tip: Download SqueezeNet 1.0 to compare models:")
        print("   python -m utils.model_downloader --model squeezenet1.0")
    
    print("\n" + "="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

