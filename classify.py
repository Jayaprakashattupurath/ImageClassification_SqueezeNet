#!/usr/bin/env python3
"""
SqueezeNet Image Classification CLI

Command-line interface for classifying images using SqueezeNet models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.squeezenet_classifier import SqueezeNetClassifier
from utils.image_utils import draw_predictions, get_image_info, validate_image


def print_predictions(image_path: str, predictions: List, show_info: bool = False):
    """Print classification results in a formatted way."""
    print("\n" + "="*70)
    print(f"Image: {os.path.basename(image_path)}")
    print("="*70)
    
    if show_info:
        info = get_image_info(image_path)
        print(f"Format: {info['format']} | Size: {info['width']}x{info['height']} | "
              f"File Size: {info['file_size_mb']:.2f} MB")
        print("-"*70)
    
    print("\nTop Predictions:")
    for i, (class_name, prob) in enumerate(predictions, 1):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{i}. {class_name:30s} {prob*100:6.2f}% {bar}")
    
    print("="*70)


def classify_single_image(args):
    """Classify a single image."""
    # Validate image
    if not validate_image(args.image):
        print(f"✗ Error: Invalid image file: {args.image}")
        return 1
    
    # Initialize classifier
    print("\n🔄 Loading SqueezeNet model...")
    classifier = SqueezeNetClassifier(args.model, args.labels)
    
    # Classify
    print(f"\n🔍 Classifying image: {args.image}")
    predictions = classifier.predict(args.image, top_k=args.top_k)
    
    # Display results
    print_predictions(args.image, predictions, show_info=args.info)
    
    # Save annotated image
    if args.output or args.save:
        output_path = args.output if args.output else f"output_{os.path.basename(args.image)}"
        annotated_path = draw_predictions(args.image, predictions, output_path)
        print(f"\n✓ Saved annotated image to: {annotated_path}")
    
    return 0


def classify_batch(args):
    """Classify multiple images."""
    # Get image files
    image_dir = Path(args.batch_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [
        str(f) for f in image_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"✗ No valid images found in: {args.batch_dir}")
        return 1
    
    print(f"\n📁 Found {len(image_files)} images in {args.batch_dir}")
    
    # Initialize classifier
    print("\n🔄 Loading SqueezeNet model...")
    classifier = SqueezeNetClassifier(args.model, args.labels)
    
    # Classify each image
    results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
        
        try:
            predictions = classifier.predict(image_path, top_k=args.top_k)
            results.append((image_path, predictions))
            
            # Print top prediction
            top_class, top_prob = predictions[0]
            print(f"  → {top_class} ({top_prob*100:.1f}%)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append((image_path, []))
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH CLASSIFICATION SUMMARY")
    print("="*70)
    for image_path, predictions in results:
        if predictions:
            top_class, top_prob = predictions[0]
            print(f"{os.path.basename(image_path):40s} → {top_class} ({top_prob*100:.1f}%)")
        else:
            print(f"{os.path.basename(image_path):40s} → Failed")
    print("="*70)
    
    return 0


def show_model_info(args):
    """Display model information."""
    print("\n🔄 Loading model...")
    classifier = SqueezeNetClassifier(args.model, args.labels)
    
    info = classifier.get_model_info()
    
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    print(f"Model Path:      {args.model}")
    print(f"Model Size:      {os.path.getsize(args.model) / (1024*1024):.2f} MB")
    print(f"Input Name:      {info['input_name']}")
    print(f"Input Shape:     {info['input_shape']}")
    print(f"Input Type:      {info['input_type']}")
    print(f"Output Name:     {info['output_name']}")
    print(f"Output Shape:    {info['output_shape']}")
    print(f"Output Type:     {info['output_type']}")
    print(f"Number of Classes: {info['num_classes']}")
    print("="*70)
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='SqueezeNet Image Classification CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single image
  python classify.py --image cat.jpg
  
  # Classify with custom model and save output
  python classify.py --image dog.jpg --model models/squeezenet1.0-12.onnx --save
  
  # Classify multiple images in a directory
  python classify.py --batch-dir images/
  
  # Show model information
  python classify.py --model-info
  
  # Get top 10 predictions
  python classify.py --image test.jpg --top-k 10
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', help='Path to input image')
    input_group.add_argument('--batch-dir', '-b', help='Directory containing images for batch processing')
    input_group.add_argument('--model-info', action='store_true', help='Show model information')
    
    # Model options
    parser.add_argument(
        '--model', '-m',
        default='models/squeezenet1.1-7.onnx',
        help='Path to ONNX model file (default: models/squeezenet1.1-7.onnx)'
    )
    parser.add_argument(
        '--labels', '-l',
        default='models/imagenet_labels.json',
        help='Path to labels JSON file (default: models/imagenet_labels.json)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Path to save annotated output image'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save annotated output image with default name'
    )
    
    # Classification options
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show image information'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not args.model_info and not os.path.exists(args.model):
        print(f"\n✗ Error: Model file not found: {args.model}")
        print("\n💡 Tip: Download the model first using:")
        print("   python setup.py")
        print("\n   Or specify a different model path with --model")
        return 1
    
    # Route to appropriate function
    try:
        if args.model_info:
            return show_model_info(args)
        elif args.batch_dir:
            return classify_batch(args)
        else:
            return classify_single_image(args)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

