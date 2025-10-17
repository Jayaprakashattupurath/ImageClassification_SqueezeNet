#!/usr/bin/env python3
"""
Basic Usage Example for SqueezeNet Classifier

This example demonstrates the basic usage of the SqueezeNet classifier.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.squeezenet_classifier import SqueezeNetClassifier


def main():
    """Basic classification example."""
    
    # Model and labels paths
    model_path = 'models/squeezenet1.1-7.onnx'
    labels_path = 'models/imagenet_labels.json'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run 'python setup.py' first.")
        return 1
    
    # Initialize classifier
    print("Loading SqueezeNet classifier...")
    classifier = SqueezeNetClassifier(model_path, labels_path)
    
    # Example image path (replace with your own)
    image_path = 'images/cat.jpg'
    
    if not os.path.exists(image_path):
        print(f"\nWarning: Example image not found: {image_path}")
        print("Please provide your own image path or download sample images.")
        return 1
    
    # Classify image
    print(f"\nClassifying image: {image_path}")
    predictions = classifier.predict(image_path, top_k=5)
    
    # Display results
    print("\n" + "="*50)
    print("Top 5 Predictions:")
    print("="*50)
    for i, (class_name, probability) in enumerate(predictions, 1):
        print(f"{i}. {class_name:30s} {probability*100:.2f}%")
    print("="*50)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

