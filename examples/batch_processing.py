#!/usr/bin/env python3
"""
Batch Processing Example

This example demonstrates how to classify multiple images efficiently.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.squeezenet_classifier import SqueezeNetClassifier
from utils.image_utils import draw_predictions


def main():
    """Batch processing example."""
    
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
    
    # Directory containing images
    images_dir = 'images'
    output_dir = 'outputs'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = []
    
    if os.path.exists(images_dir):
        image_files = [
            str(f) for f in Path(images_dir).iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
    
    if not image_files:
        print(f"\nNo images found in '{images_dir}' directory.")
        print("Please add some images to process.")
        return 1
    
    print(f"\nFound {len(image_files)} images to process")
    print("="*60)
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n[{i}/{len(image_files)}] Processing: {image_name}")
        
        try:
            # Classify
            predictions = classifier.predict(image_path, top_k=3)
            
            # Display top prediction
            top_class, top_prob = predictions[0]
            print(f"  Top prediction: {top_class} ({top_prob*100:.1f}%)")
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"output_{image_name}")
            draw_predictions(image_path, predictions, output_path)
            print(f"  Saved to: {output_path}")
            
            results.append({
                'image': image_name,
                'predictions': predictions,
                'success': True
            })
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'image': image_name,
                'predictions': [],
                'success': False
            })
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nProcessed: {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    
    print("\nResults:")
    for result in results:
        if result['success']:
            top_class, top_prob = result['predictions'][0]
            print(f"  ✓ {result['image']:30s} → {top_class} ({top_prob*100:.1f}%)")
        else:
            print(f"  ✗ {result['image']:30s} → Failed")
    
    print("="*60)
    print(f"\nAnnotated images saved to: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

