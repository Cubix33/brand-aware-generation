from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import replicate
import os
from dotenv import load_dotenv
import base64
import io
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import requests

# Load environment variables
load_dotenv()

app = FastAPI(title="Brand AI Generator API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Adobe Express domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class LogoData(BaseModel):
    name: str
    type: str
    data: str

class Colors(BaseModel):
    primary: str
    secondary: str
    accent: str

class ReferenceImage(BaseModel):
    id: float
    name: str
    data: str

class BrandData(BaseModel):
    logo: Optional[LogoData] = None
    colors: Colors
    font: str
    references: List[ReferenceImage] = []
    description: str = ""

class AudioConfig(BaseModel):
    type: str
    prompt: str

class GenerationRequest(BaseModel):
    prompt: str
    brandData: BrandData
    outputType: str
    frameCount: int = 1
    audio: Optional[AudioConfig] = None

class Asset(BaseModel):
    id: str
    type: str
    url: str
    prompt: str

class GenerationResponse(BaseModel):
    assets: List[Asset]
    success: bool
    message: str = ""

class PosterGenerationRequest(BaseModel):
    prompt: str
    referenceImages: List[str]  # List of base64 image data URIs
    logo: Optional[str] = None  # Optional base64 logo image
    productImage: Optional[str] = None  # Optional base64 product image

class PosterLayer(BaseModel):
    name: str
    data: str  # base64 data URI
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

class PosterGenerationResponse(BaseModel):
    success: bool
    message: str = ""
    composite: Optional[str] = None  # base64 data URI of final composite
    layers: Optional[Dict[str, PosterLayer]] = None
    metadata: Optional[Dict] = None

# Helper Functions
def extract_dominant_colors(logo_data: str, n_colors: int = 5) -> List[str]:
    """Extract dominant colors from logo using K-means clustering"""
    try:
        # Decode base64 image
        if ',' in logo_data:
            img_data = base64.b64decode(logo_data.split(',')[1])
        else:
            img_data = base64.b64decode(logo_data)
        
        # Open image
        img = Image.open(io.BytesIO(img_data))
        img = img.convert('RGB')
        img = img.resize((150, 150))  # Resize for faster processing
        
        # Convert to numpy array
        pixels = np.array(img).reshape(-1, 3)
        
        # Remove white/near-white pixels (background)
        pixels = pixels[~np.all(pixels > 240, axis=1)]
        
        if len(pixels) < n_colors:
            return []
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = []
        for center in kmeans.cluster_centers_:
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(center[0]), 
                int(center[1]), 
                int(center[2])
            )
            colors.append(hex_color)
        
        return colors
    
    except Exception as e:
        print(f"Error extracting colors: {e}")
        return []

def build_enhanced_prompt(
    user_prompt: str,
    brand_colors: Colors,
    brand_description: str,
    logo_colors: List[str]
) -> str:
    """Build an enhanced prompt with brand context"""
    
    # Base prompt
    enhanced = user_prompt
    
    # Add brand style
    if brand_description:
        enhanced += f", {brand_description} style"
    
    # Add color guidance
    color_description = f"using brand colors {brand_colors.primary}, {brand_colors.accent}"
    if logo_colors:
        color_description += f", incorporating {', '.join(logo_colors[:3])}"
    
    enhanced += f", {color_description}"
    
    # Add quality modifiers
    enhanced += ", professional design, high quality, clean composition, modern aesthetic"
    
    return enhanced

def download_image_as_base64(url: str) -> str:
    """Download image from URL and convert to base64 data URI"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Detect image type
        content_type = response.headers.get('content-type', 'image/png')
        
        # Convert to base64
        img_base64 = base64.b64encode(response.content).decode('utf-8')
        
        return f"data:{content_type};base64,{img_base64}"
    
    except Exception as e:
        print(f"Error downloading image: {e}")
        raise

def generate_image_with_replicate(prompt: str, negative_prompt: str = None) -> str:
    """Generate image using Replicate's SDXL model"""
    
    if not os.getenv("REPLICATE_API_TOKEN"):
        raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
    
    # Default negative prompt
    if negative_prompt is None:
        negative_prompt = "blurry, low quality, distorted, watermark, text, logo, signature, bad composition, ugly"
    
    print(f"🎨 Generating with SDXL...")
    print(f"📝 Prompt: {prompt}")
    
    try:
        # Use SDXL model
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": 1024,
                "height": 1024,
                "num_outputs": 1,
                "num_inference_steps": 40,  # Higher = better quality
                "guidance_scale": 7.5,
                "scheduler": "DPMSolverMultistep",
            }
        )
        
        # Output is a list of URLs
        image_url = output[0]
        print(f"✅ Generated image: {image_url}")
        
        # Download and convert to base64 (so it works in Adobe Express)
        base64_image = download_image_as_base64(image_url)
        
        return base64_image
    
    except Exception as e:
        print(f"❌ Replicate generation error: {e}")
        raise

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Brand AI Generator API",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    has_token = bool(os.getenv("REPLICATE_API_TOKEN"))
    return {
        "status": "healthy",
        "replicate_configured": has_token
    }

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """Main generation endpoint"""
    
    try:
        print("\n" + "="*50)
        print("🚀 GENERATION REQUEST RECEIVED")
        print("="*50)
        print(f"📝 User Prompt: {request.prompt}")
        print(f"🎨 Output Type: {request.outputType}")
        print(f"🔢 Frame Count: {request.frameCount}")
        print(f"🎨 Brand Colors: {request.brandData.colors}")
        
        # Extract logo colors if available
        logo_colors = []
        if request.brandData.logo:
            print("🔍 Extracting colors from logo...")
            logo_colors = extract_dominant_colors(request.brandData.logo.data)
            print(f"✅ Extracted colors: {logo_colors}")
        
        # Build enhanced prompt
        enhanced_prompt = build_enhanced_prompt(
            request.prompt,
            request.brandData.colors,
            request.brandData.description,
            logo_colors
        )
        
        print(f"\n✨ Enhanced Prompt: {enhanced_prompt}\n")
        
        # Generate assets
        assets = []
        
        for i in range(request.frameCount):
            print(f"\n📸 Generating frame {i+1}/{request.frameCount}...")
            
            # Add variation to prompt for multiple frames
            frame_prompt = enhanced_prompt
            if request.frameCount > 1:
                variations = [
                    "dynamic composition",
                    "alternate perspective",
                    "different layout",
                    "creative angle",
                    "unique arrangement"
                ]
                frame_prompt += f", {variations[i % len(variations)]}"
            
            # Generate image
            image_base64 = generate_image_with_replicate(frame_prompt)
            
            asset = Asset(
                id=f"asset_{i}_{int(np.random.random() * 10000)}",
                type="image",
                url=image_base64,
                prompt=frame_prompt
            )
            
            assets.append(asset)
            print(f"✅ Frame {i+1} generated successfully")
        
        print("\n" + "="*50)
        print(f"🎉 GENERATION COMPLETE - {len(assets)} assets created")
        print("="*50 + "\n")
        
        return GenerationResponse(
            assets=assets,
            success=True,
            message=f"Successfully generated {len(assets)} brand-consistent asset(s)"
        )
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR: {error_msg}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {error_msg}"
        )

@app.post("/api/test-generation")
async def test_generation():
    """Test endpoint to verify Replicate is working"""
    try:
        print("🧪 Testing Replicate connection...")
        
        test_prompt = "A beautiful sunset over mountains, professional photography"
        image_url = generate_image_with_replicate(test_prompt)
        
        return {
            "success": True,
            "message": "Replicate is working!",
            "test_image": image_url[:100] + "..."  # Preview
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )

@app.post("/api/generate-poster", response_model=PosterGenerationResponse)
async def generate_poster(request: PosterGenerationRequest):
    """
    Generate a marketing poster using BrandDiffusion
    
    This endpoint accepts:
    - prompt: User's text prompt describing the poster
    - referenceImages: List of base64-encoded reference images (at least one required)
    - logo: Optional base64-encoded logo image
    - productImage: Optional base64-encoded product image
    """
    try:
        print("\n" + "="*50)
        print("🎨 POSTER GENERATION REQUEST RECEIVED")
        print("="*50)
        print(f"📝 User Prompt: {request.prompt}")
        print(f"🖼️  Reference Images: {len(request.referenceImages)}")
        print(f"🏷️  Logo: {'✅ Provided' if request.logo else '❌ Not provided'}")
        print(f"📦 Product Image: {'✅ Provided' if request.productImage else '❌ Not provided'}")
        
        # Validate input
        if not request.referenceImages or len(request.referenceImages) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one reference image is required"
            )
        
        # Try to import and use BrandDiffusion
        try:
            from branddiffusion_wrapper import generate_poster_from_base64
            
            # Generate poster
            result = generate_poster_from_base64(
                prompt=request.prompt,
                reference_images=request.referenceImages,
                logo_data=request.logo,
                product_image_data=request.productImage
            )
            
            # Convert layers format to match response model
            layers_dict = None
            if result.get("layers"):
                layers_dict = {}
                for name, layer_data in result["layers"].items():
                    layers_dict[name] = PosterLayer(
                        name=name,
                        data=layer_data.get("data", ""),
                        x=layer_data.get("x"),
                        y=layer_data.get("y"),
                        width=layer_data.get("width"),
                        height=layer_data.get("height")
                    )
            
            return PosterGenerationResponse(
                success=result.get("success", False),
                message=result.get("message", "Poster generated successfully"),
                composite=result.get("composite"),
                layers=layers_dict,
                metadata=result.get("metadata", {})
            )
        
        except ImportError as e:
            print(f"❌ BrandDiffusion dependencies not available: {e}")
            return PosterGenerationResponse(
                success=False,
                message=f"BrandDiffusion dependencies not installed: {str(e)}",
                composite=None,
                layers=None,
                metadata={"error": "missing_dependencies"}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ POSTER GENERATION ERROR: {error_msg}\n")
        raise HTTPException(
            status_code=500,
            detail=f"Poster generation failed: {error_msg}"
        )

# Run server
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("🚀 Brand AI Generator Backend Starting...")
    print("="*50)
    print(f"📡 API will be available at: http://localhost:8000")
    print(f"📚 Docs available at: http://localhost:8000/docs")
    print(f"🔑 Replicate Token: {'✅ Configured' if os.getenv('REPLICATE_API_TOKEN') else '❌ Missing'}")
    print("="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
