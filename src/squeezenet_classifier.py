"""
SqueezeNet Image Classifier Module

This module provides the core functionality for image classification using SqueezeNet models.
It supports both SqueezeNet 1.0 and 1.1 variants with ONNX Runtime inference.
"""

import os
import numpy as np
from PIL import Image
import onnxruntime as ort
from typing import List, Tuple, Optional
import json


class SqueezeNetClassifier:
    """
    SqueezeNet Image Classifier
    
    A classifier that uses pre-trained SqueezeNet models to perform image classification
    on ImageNet-1000 classes.
    """
    
    def __init__(self, model_path: str, labels_path: Optional[str] = None):
        """
        Initialize the SqueezeNet classifier.
        
        Args:
            model_path: Path to the ONNX model file
            labels_path: Path to the ImageNet labels JSON file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load labels
        self.labels = self._load_labels(labels_path)
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"✓ Model loaded successfully: {os.path.basename(model_path)}")
        print(f"✓ Input name: {self.input_name}")
        print(f"✓ Output name: {self.output_name}")
    
    def _load_labels(self, labels_path: Optional[str]) -> List[str]:
        """Load ImageNet class labels."""
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            print(f"✓ Loaded {len(labels)} class labels")
            return labels
        else:
            # Default labels (indices only)
            print("⚠ Using default numeric labels (0-999)")
            return [f"class_{i}" for i in range(1000)]
    
    def preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess an image for SqueezeNet inference.
        
        Args:
            image_path: Path to the input image
            target_size: Target size for resizing (height, width)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Apply ImageNet normalization
        img_array = (img_array - self.mean) / self.std
        
        # Convert from HWC to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify an image and return top-k predictions.
        
        Args:
            image_path: Path to the input image
            top_k: Number of top predictions to return
            
        Returns:
            List of (class_name, probability) tuples
        """
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Get probabilities
        logits = outputs[0][0]
        probabilities = self.softmax(logits)
        
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.labels[idx]
            probability = float(probabilities[idx])
            results.append((class_name, probability))
        
        return results
    
    def predict_batch(self, image_paths: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of paths to input images
            top_k: Number of top predictions to return per image
            
        Returns:
            List of prediction results for each image
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, top_k)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append([])
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]
        
        return {
            'input_name': input_info.name,
            'input_shape': input_info.shape,
            'input_type': input_info.type,
            'output_name': output_info.name,
            'output_shape': output_info.shape,
            'output_type': output_info.type,
            'num_classes': len(self.labels)
        }

