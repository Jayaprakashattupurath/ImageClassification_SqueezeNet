# Sample Images Directory

This directory is for storing test images for classification.

## Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- GIF
- WebP

## Getting Sample Images

You can add your own images here, or download sample images from:

### Free Image Sources

1. **Unsplash** - https://unsplash.com/
   - High-quality free images
   - No attribution required

2. **Pexels** - https://www.pexels.com/
   - Free stock photos and videos
   - CC0 license

3. **Pixabay** - https://pixabay.com/
   - Free images and videos
   - CC0 license

4. **ImageNet Sample Images**
   - You can search for ImageNet sample images online
   - Useful for testing since the model is trained on ImageNet

## Example Images for Testing

For best results with SqueezeNet, try images containing:

- Animals (cats, dogs, birds, elephants, etc.)
- Vehicles (cars, airplanes, bicycles)
- Common objects (furniture, electronics, tools)
- Food items (fruits, dishes)
- Nature scenes (landscapes, flowers)

## Usage

Once you have images in this directory:

```bash
# Classify a single image
python classify.py --image images/your_image.jpg

# Process all images in this directory
python classify.py --batch-dir images/

# Use in web interface
python app.py
# Then upload images through the web UI
```

## Note

The `.gitignore` file is configured to ignore large image files by default to keep the repository size manageable. You can modify `.gitignore` if you want to commit specific images to the repository.

