"""
Image Utility Functions

Helper functions for image processing and visualization.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_predictions(
    image_path: str,
    predictions: List[Tuple[str, float]],
    output_path: str = None,
    show: bool = False
) -> str:
    """
    Draw predictions on an image.
    
    Args:
        image_path: Path to input image
        predictions: List of (class_name, probability) tuples
        output_path: Path to save annotated image (optional)
        show: Whether to display the image
        
    Returns:
        Path to the output image
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
        small_font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Prepare text
    text_lines = ["Top Predictions:"]
    for i, (class_name, prob) in enumerate(predictions[:5], 1):
        text_lines.append(f"{i}. {class_name}: {prob*100:.1f}%")
    
    # Calculate text box dimensions
    max_width = max([draw.textlength(line, font=small_font) for line in text_lines])
    line_height = 25
    text_height = len(text_lines) * line_height + 20
    
    # Draw semi-transparent background
    padding = 10
    box_coords = [padding, padding, max_width + padding * 3, text_height + padding]
    
    # Create a new image with transparency for the overlay
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(box_coords, fill=(0, 0, 0, 200))
    
    # Composite the overlay
    img = img.convert('RGBA')
    img = Image.alpha_composite(img, overlay)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Draw text
    y_offset = padding * 2
    for i, line in enumerate(text_lines):
        color = (255, 255, 0) if i == 0 else (255, 255, 255)  # Yellow for title
        draw.text((padding * 2, y_offset), line, fill=color, font=small_font)
        y_offset += line_height
    
    # Save output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output_{base_name}.jpg"
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    img.save(output_path, quality=95)
    
    if show:
        img.show()
    
    return output_path


def validate_image(image_path: str) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get information about an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    img = Image.open(image_path)
    
    return {
        'path': image_path,
        'format': img.format,
        'mode': img.mode,
        'size': img.size,
        'width': img.width,
        'height': img.height,
        'file_size_mb': os.path.getsize(image_path) / (1024 * 1024)
    }


def resize_image(image_path: str, max_size: int = 1024, output_path: str = None) -> str:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image_path: Path to input image
        max_size: Maximum dimension (width or height)
        output_path: Path to save resized image
        
    Returns:
        Path to the resized image
    """
    img = Image.open(image_path)
    
    # Calculate new dimensions
    ratio = min(max_size / img.width, max_size / img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # Save
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1]
        output_path = f"{base_name}_resized{ext}"
    
    img.save(output_path, quality=95)
    return output_path


def create_image_grid(image_paths: List[str], grid_size: Tuple[int, int] = None) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        image_paths: List of image paths
        grid_size: (rows, cols) for the grid. Auto-calculated if None
        
    Returns:
        PIL Image object with the grid
    """
    if not image_paths:
        raise ValueError("No images provided")
    
    # Load images
    images = [Image.open(path).convert('RGB') for path in image_paths]
    
    # Determine grid size
    n_images = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Find max dimensions
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    
    # Resize all images to max dimensions
    resized_images = []
    for img in images:
        new_img = Image.new('RGB', (max_width, max_height), (255, 255, 255))
        offset_x = (max_width - img.width) // 2
        offset_y = (max_height - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))
        resized_images.append(new_img)
    
    # Create grid
    grid_width = cols * max_width
    grid_height = rows * max_height
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height
        grid.paste(img, (x, y))
    
    return grid

