"""
Simplified API wrapper for BrandDiffusion integration with FastAPI
This module provides functions to run BrandDiffusion generation from API requests
"""
import os
import base64
import io
import tempfile
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, List, Tuple

def image_base64_to_pil(base64_data: str) -> Image.Image:
    """Convert base64 data URI to PIL Image"""
    if ',' in base64_data:
        header, data = base64_data.split(',', 1)
    else:
        data = base64_data
    
    image_data = base64.b64decode(data)
    return Image.open(io.BytesIO(image_data)).convert("RGB")

def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data URI"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_base64}"

def save_temp_image(image: Image.Image, suffix: str = ".png") -> str:
    """Save PIL Image to temporary file and return path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    image.save(temp_file.name)
    return temp_file.name

def generate_poster_from_api(
    user_prompt: str,
    reference_images: List[str],  # List of base64 data URIs
    logo_data: Optional[str] = None,  # Optional base64 logo
    product_image_data: Optional[str] = None,  # Optional base64 product image
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate a marketing poster using BrandDiffusion
    
    Args:
        user_prompt: User's text prompt
        reference_images: List of base64-encoded reference images
        logo_data: Optional base64-encoded logo image
        product_image_data: Optional base64-encoded product image
        output_dir: Optional output directory (defaults to temp)
    
    Returns:
        Dictionary with:
        - composite: base64-encoded final poster
        - layers: dict of layer_name -> base64-encoded image
        - metadata: generation metadata
    """
    
    # Try to import BrandDiffusion (optional dependency)
    try:
        from branddiffusion import main as branddiffusion_main
        HAS_BRANDDIFFUSION = True
    except ImportError:
        HAS_BRANDDIFFUSION = False
        print("⚠️ BrandDiffusion not available - missing dependencies")
    
    if not HAS_BRANDDIFFUSION:
        raise RuntimeError(
            "BrandDiffusion dependencies not installed. "
            "Please install: torch, diffusers, transformers, mediapipe, gfpgan"
        )
    
    # Create temporary directory for outputs
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="branddiffusion_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert base64 images to temporary files
    temp_files = []
    ref_image_path = None
    logo_path = None
    product_image_path = None
    
    try:
        # Process reference image (use first one)
        if reference_images and len(reference_images) > 0:
            ref_image = image_base64_to_pil(reference_images[0])
            ref_image_path = save_temp_image(ref_image)
            temp_files.append(ref_image_path)
        
        # Process logo if provided
        if logo_data:
            logo_image = image_base64_to_pil(logo_data)
            logo_path = save_temp_image(logo_image)
            temp_files.append(logo_path)
        
        # Process product image if provided
        if product_image_data:
            product_image = image_base64_to_pil(product_image_data)
            product_image_path = save_temp_image(product_image)
            temp_files.append(product_image_path)
        
        # Call BrandDiffusion main function
        # Note: This assumes BrandDiffusion has been refactored to accept these params
        # We'll need to create a wrapper function in branddiffusion.py
        
        # For now, return a placeholder response structure
        # TODO: Integrate with actual BrandDiffusion execution
        
        return {
            "success": False,
            "message": "BrandDiffusion integration in progress - please use the original branddiffusion.py script for now",
            "output_dir": output_dir
        }
    
    finally:
        # Cleanup temp files (optional - keep for debugging)
        # for temp_file in temp_files:
        #     try:
        #         os.unlink(temp_file)
        #     except:
        #         pass
        pass
