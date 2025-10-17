#!/usr/bin/env python3
"""
SqueezeNet Image Classification Web Application

A Gradio-based web interface for interactive image classification.
"""

import os
import sys
import gradio as gr
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.squeezenet_classifier import SqueezeNetClassifier
from utils.image_utils import get_image_info


# Global classifier instance
classifier = None


def load_model(model_path='models/squeezenet1.1-7.onnx', labels_path='models/imagenet_labels.json'):
    """Load the SqueezeNet model."""
    global classifier
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Please run 'python setup.py' first to download the model."
        )
    
    print("Loading SqueezeNet model...")
    classifier = SqueezeNetClassifier(model_path, labels_path)
    print("Model loaded successfully!")


def classify_image(image):
    """
    Classify an image using SqueezeNet.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Dictionary of class probabilities for Gradio
    """
    if classifier is None:
        return {"Error": 1.0}
    
    try:
        # Save temporary image if numpy array
        if isinstance(image, np.ndarray):
            temp_path = "temp_input.jpg"
            Image.fromarray(image).save(temp_path)
        else:
            temp_path = "temp_input.jpg"
            image.save(temp_path)
        
        # Get predictions
        predictions = classifier.predict(temp_path, top_k=10)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Convert to dictionary for Gradio
        result = {class_name: float(prob) for class_name, prob in predictions}
        
        return result
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return {"Error": 1.0, "Details": str(e)[:100]}


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    # Create interface
    with gr.Blocks(css=custom_css, title="SqueezeNet Image Classifier") as demo:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1>🖼️ SqueezeNet Image Classification</h1>
                <p>Upload an image to classify it into one of 1000 ImageNet categories</p>
            </div>
        """)
        
        # Description
        gr.Markdown("""
        ### About SqueezeNet
        SqueezeNet is a compact convolutional neural network that achieves AlexNet-level accuracy 
        with 50x fewer parameters and <0.5MB model size. It's ideal for mobile and embedded applications.
        
        **Model:** SqueezeNet 1.1  
        **Dataset:** ImageNet (1000 classes)  
        **Input Size:** 224x224 pixels  
        """)
        
        # Main interface
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"]
                )
                
                classify_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")
                
                gr.Markdown("### 📸 Try Sample Images")
                gr.Examples(
                    examples=[
                        "images/cat.jpg",
                        "images/dog.jpg",
                        "images/car.jpg"
                    ] if all(os.path.exists(f"images/{img}.jpg") for img in ["cat", "dog", "car"]) else [],
                    inputs=image_input,
                    label="Sample Images"
                )
            
            with gr.Column(scale=1):
                output_labels = gr.Label(
                    label="Classification Results",
                    num_top_classes=10
                )
                
                gr.Markdown("""
                ### 🎯 Understanding Results
                - **Confidence scores** range from 0% to 100%
                - Higher scores indicate stronger confidence
                - Multiple objects may appear with varying confidence
                """)
        
        # Model Information
        with gr.Accordion("ℹ️ Model Information", open=False):
            if classifier:
                info = classifier.get_model_info()
                gr.Markdown(f"""
                **Model Details:**
                - Input Shape: {info['input_shape']}
                - Output Shape: {info['output_shape']}
                - Number of Classes: {info['num_classes']}
                - Model Size: {os.path.getsize('models/squeezenet1.1-7.onnx') / (1024*1024):.2f} MB
                """)
            else:
                gr.Markdown("*Model information not available*")
        
        # Connect components
        classify_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=output_labels
        )
        
        image_input.change(
            fn=classify_image,
            inputs=image_input,
            outputs=output_labels
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📚 Resources
        - [SqueezeNet Paper](https://arxiv.org/abs/1602.07360)
        - [ONNX Model Zoo](https://github.com/onnx/models)
        - [ImageNet Classes](http://www.image-net.org/)
        """)
    
    return demo


def main():
    """Launch the web application."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SqueezeNet Web Application')
    parser.add_argument(
        '--model',
        default='models/squeezenet1.1-7.onnx',
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--labels',
        default='models/imagenet_labels.json',
        help='Path to labels JSON file'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host address (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port number (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public shareable link'
    )
    
    args = parser.parse_args()
    
    try:
        # Load model
        load_model(args.model, args.labels)
        
        # Create and launch interface
        demo = create_interface()
        
        print("\n" + "="*70)
        print("🚀 Starting SqueezeNet Web Application")
        print("="*70)
        print(f"\n📍 Local URL: http://{args.host}:{args.port}")
        if args.share:
            print("🌐 Creating public URL...")
        print("\n💡 Press Ctrl+C to stop the server\n")
        
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {str(e)}")
        print("\n💡 Please run 'python setup.py' first to download the model.")
        return 1
    except Exception as e:
        print(f"\n✗ Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

