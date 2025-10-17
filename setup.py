#!/usr/bin/env python3
"""
Setup Script for SqueezeNet Application

Downloads required models and prepares the environment.
"""

import sys
import os

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.model_downloader import ModelDownloader


def main():
    """Main setup function."""
    print("\n" + "="*70)
    print("SqueezeNet Image Classification - Setup")
    print("="*70)
    print("\nThis script will download:")
    print("  • SqueezeNet 1.1 model (~5 MB)")
    print("  • ImageNet class labels (~100 KB)")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Setup cancelled.")
        return 0
    
    # Download models
    try:
        downloader = ModelDownloader('models')
        model_path, labels_path = downloader.setup_all('squeezenet1.1')
        
        print("\n" + "="*70)
        print("✓ Setup completed successfully!")
        print("="*70)
        print("\nYou can now classify images using:")
        print("  python classify.py --image path/to/image.jpg")
        print("\nOr start the web interface:")
        print("  python app.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

