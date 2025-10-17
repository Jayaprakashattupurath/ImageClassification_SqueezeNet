"""
Model Downloader Utility

Downloads SqueezeNet ONNX models and ImageNet labels from official sources.
"""

import os
import requests
from tqdm import tqdm
import json


class ModelDownloader:
    """Download and setup SqueezeNet models and labels."""
    
    # Official ONNX model zoo URLs
    MODELS = {
        'squeezenet1.0': 'https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx',
        'squeezenet1.1': 'https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx'
    }
    
    # ImageNet labels
    LABELS_URL = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the model downloader.
        
        Args:
            models_dir: Directory to save models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def download_file(self, url: str, destination: str, description: str = "Downloading"):
        """
        Download a file with progress bar.
        
        Args:
            url: URL to download from
            destination: Path to save the file
            description: Description for the progress bar
        """
        print(f"\n{description}...")
        print(f"URL: {url}")
        print(f"Destination: {destination}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=description
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Successfully downloaded to: {destination}")
            return True
            
        except Exception as e:
            print(f"✗ Error downloading file: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
            return False
    
    def download_model(self, model_name: str = 'squeezenet1.1') -> str:
        """
        Download a SqueezeNet model.
        
        Args:
            model_name: Name of the model ('squeezenet1.0' or 'squeezenet1.1')
            
        Returns:
            Path to the downloaded model
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        # Determine filename from URL
        url = self.MODELS[model_name]
        filename = url.split('/')[-1]
        destination = os.path.join(self.models_dir, filename)
        
        # Check if already downloaded
        if os.path.exists(destination):
            print(f"✓ Model already exists: {destination}")
            return destination
        
        # Download
        success = self.download_file(
            url, 
            destination, 
            f"Downloading {model_name}"
        )
        
        if success:
            return destination
        else:
            raise RuntimeError(f"Failed to download {model_name}")
    
    def download_labels(self) -> str:
        """
        Download ImageNet class labels.
        
        Returns:
            Path to the labels file
        """
        destination = os.path.join(self.models_dir, 'imagenet_labels.json')
        
        # Check if already downloaded
        if os.path.exists(destination):
            print(f"✓ Labels already exist: {destination}")
            return destination
        
        # Download
        success = self.download_file(
            self.LABELS_URL,
            destination,
            "Downloading ImageNet labels"
        )
        
        if success:
            return destination
        else:
            raise RuntimeError("Failed to download labels")
    
    def setup_all(self, model_name: str = 'squeezenet1.1'):
        """
        Download model and labels.
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            Tuple of (model_path, labels_path)
        """
        print("\n" + "="*60)
        print("SqueezeNet Model Setup")
        print("="*60)
        
        model_path = self.download_model(model_name)
        labels_path = self.download_labels()
        
        print("\n" + "="*60)
        print("Setup Complete!")
        print("="*60)
        print(f"Model: {model_path}")
        print(f"Labels: {labels_path}")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        return model_path, labels_path


def main():
    """Command-line interface for model downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download SqueezeNet models')
    parser.add_argument(
        '--model',
        choices=['squeezenet1.0', 'squeezenet1.1'],
        default='squeezenet1.1',
        help='Model version to download'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory to save models'
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    downloader.setup_all(args.model)


if __name__ == '__main__':
    main()

