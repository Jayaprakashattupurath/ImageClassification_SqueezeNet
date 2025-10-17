#!/usr/bin/env python3
"""
Installation Test Script

Run this script to verify that everything is set up correctly.
"""

import sys
import os


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 7:
        print(f"  ✓ Python {major}.{minor} (OK)")
        return True
    else:
        print(f"  ✗ Python {major}.{minor} (Need 3.7+)")
        return False


def check_dependencies():
    """Check if all required packages are installed."""
    print("\nChecking dependencies...")
    
    packages = {
        'onnxruntime': 'onnxruntime',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'gradio': 'gradio',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} (missing)")
            missing.append(package_name)
    
    return len(missing) == 0, missing


def check_models():
    """Check if models are downloaded."""
    print("\nChecking models...")
    
    model_path = 'models/squeezenet1.1-7.onnx'
    labels_path = 'models/imagenet_labels.json'
    
    model_ok = os.path.exists(model_path)
    labels_ok = os.path.exists(labels_path)
    
    if model_ok:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ SqueezeNet model ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ SqueezeNet model (missing)")
    
    if labels_ok:
        print(f"  ✓ ImageNet labels")
    else:
        print(f"  ✗ ImageNet labels (missing)")
    
    return model_ok and labels_ok


def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_paths = [
        'src/squeezenet_classifier.py',
        'utils/model_downloader.py',
        'utils/image_utils.py',
        'classify.py',
        'app.py',
        'setup.py'
    ]
    
    all_ok = True
    for path in required_paths:
        if os.path.exists(path):
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (missing)")
            all_ok = False
    
    return all_ok


def test_import():
    """Test importing the main module."""
    print("\nTesting module imports...")
    
    try:
        from src.squeezenet_classifier import SqueezeNetClassifier
        print("  ✓ SqueezeNetClassifier imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to import: {str(e)}")
        return False


def test_classifier():
    """Test basic classifier functionality."""
    print("\nTesting classifier (if model available)...")
    
    model_path = 'models/squeezenet1.1-7.onnx'
    labels_path = 'models/imagenet_labels.json'
    
    if not os.path.exists(model_path):
        print("  ⚠ Skipped (model not downloaded)")
        return True
    
    try:
        from src.squeezenet_classifier import SqueezeNetClassifier
        
        classifier = SqueezeNetClassifier(model_path, labels_path)
        info = classifier.get_model_info()
        
        print(f"  ✓ Model loaded successfully")
        print(f"    - Input shape: {info['input_shape']}")
        print(f"    - Output shape: {info['output_shape']}")
        print(f"    - Classes: {info['num_classes']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("SqueezeNet Image Classification - Installation Test")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Python version", check_python_version()))
    
    deps_ok, missing = check_dependencies()
    results.append(("Dependencies", deps_ok))
    
    results.append(("Project structure", check_structure()))
    results.append(("Module imports", test_import()))
    results.append(("Models", check_models()))
    results.append(("Classifier", test_classifier()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Try the web interface: python app.py")
        print("  2. Try the CLI: python classify.py --image your_image.jpg")
        print("  3. Check examples: python examples/basic_usage.py")
        print("\nSee QUICKSTART.md for more details.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
        
        if not deps_ok:
            print("\nTo install missing dependencies:")
            print("  pip install -r requirements.txt")
        
        if not results[4][1]:  # Models not found
            print("\nTo download models:")
            print("  python setup.py")
        
        return 1


if __name__ == '__main__':
    sys.exit(main())

