"""
Wrapper to integrate BrandDiffusion with FastAPI
This module provides a clean API interface for BrandDiffusion poster generation
"""
import os
import io
import base64
import tempfile
import json
from typing import Dict, Optional, List
from PIL import Image

# Import helper functions from branddiffusion_api
from branddiffusion_api import image_base64_to_pil, pil_to_base64, save_temp_image

def generate_poster(
    prompt: str,
    reference_image_path: str,
    logo_path: Optional[str] = None,
    product_image_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate a marketing poster using BrandDiffusion (refactored version)
    
    This function integrates the BrandDiffusion pipeline and returns results as a dictionary
    instead of saving to disk.
    
    Args:
        prompt: User's text prompt
        reference_image_path: Path to reference image file
        logo_path: Optional path to logo image file
        product_image_path: Optional path to product image file
        output_dir: Optional output directory (for temporary files)
    
    Returns:
        Dictionary with:
        - composite: PIL Image of final composite
        - layers: dict of layer_name -> PIL Image
        - metadata: generation metadata (copy, use_case, etc.)
    """
    
    # Import BrandDiffusion generate_poster function
    try:
        from branddiffusion import generate_poster as bd_generate_poster
    except ImportError as e:
        raise RuntimeError(f"BrandDiffusion dependencies not available: {e}")
    
    # Load images from file paths
    ref_images = [Image.open(reference_image_path).convert("RGB")]
    
    logo_image = None
    if logo_path:
        logo_image = Image.open(logo_path).convert("RGBA")
    
    product_image = None
    if product_image_path:
        product_image = Image.open(product_image_path).convert("RGBA")
    
    # Call BrandDiffusion generate_poster function
    result = bd_generate_poster(
        prompt=prompt,
        ref_images=ref_images,
        logo_image=logo_image,
        product_image=product_image,
        output_dir=output_dir
    )
    
    return result

def generate_poster_from_base64(
    prompt: str,
    reference_images: List[str],  # List of base64 data URIs
    logo_data: Optional[str] = None,
    product_image_data: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Generate poster from base64-encoded images (API-friendly wrapper)
    
    This function:
    1. Converts base64 images to temporary files
    2. Calls generate_poster()
    3. Converts PIL results back to base64
    4. Returns API-friendly response
    
    Returns:
        Dictionary with base64-encoded images and metadata
    """
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="branddiffusion_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    temp_files = []
    
    try:
        # Convert base64 to temporary files
        ref_image = image_base64_to_pil(reference_images[0])
        ref_path = save_temp_image(ref_image, suffix=".png")
        temp_files.append(ref_path)
        
        logo_path = None
        if logo_data:
            logo_image = image_base64_to_pil(logo_data)
            logo_path = save_temp_image(logo_image, suffix=".png")
            temp_files.append(logo_path)
        
        product_path = None
        if product_image_data:
            product_image = image_base64_to_pil(product_image_data)
            product_path = save_temp_image(product_image, suffix=".png")
            temp_files.append(product_path)
        
        # Call BrandDiffusion
        result = generate_poster(
            prompt=prompt,
            reference_image_path=ref_path,
            logo_path=logo_path,
            product_image_path=product_path,
            output_dir=output_dir
        )
        
        # Convert PIL Images to base64
        response = {
            "success": True,
            "message": "Poster generated successfully",
            "composite": None,
            "layers": {},
            "metadata": result.get("metadata", {})
        }
        
        # Convert composite
        if result.get("composite") and isinstance(result["composite"], Image.Image):
            response["composite"] = pil_to_base64(result["composite"])
        
        # Convert layers - ensure we have the layer metadata
        layers = {}
        if result.get("layers"):
            layer_meta = result.get("metadata", {}).get("layout_meta", {})
            for name, layer_img in result["layers"].items():
                if isinstance(layer_img, Image.Image):
                    layer_data = {
                        "name": name,
                        "data": pil_to_base64(layer_img)
                    }
                    # Add position/size metadata if available
                    if name in layer_meta:
                        meta = layer_meta[name]
                        layer_data["x"] = meta.get("x")
                        layer_data["y"] = meta.get("y")
                        layer_data["width"] = meta.get("width")
                        layer_data["height"] = meta.get("height")
                    layers[name] = layer_data
        response["layers"] = layers
        
        return response
    
    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "composite": None,
            "layers": None,
            "metadata": {"error": str(e)}
        }
    
    finally:
        # Optionally cleanup temp files
        # Uncomment if you want to clean up immediately
        # for temp_file in temp_files:
        #     try:
        #         os.unlink(temp_file)
        #     except:
        #         pass
        pass
