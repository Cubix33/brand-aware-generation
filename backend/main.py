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
import httpx

# Load environment variables
load_dotenv()

app = FastAPI(title="Brand AI Generator API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS
# ============================================================================

class BrandColors(BaseModel):
    primary: str
    secondary: str
    accent: str

class BrandLogo(BaseModel):
    name: str
    type: str
    data: str  # base64

class BrandReference(BaseModel):
    id: float
    name: str
    data: str  # base64

class BrandData(BaseModel):
    logo: Optional[BrandLogo]
    colors: BrandColors
    font: str
    references: List[BrandReference]
    description: str

class AudioConfig(BaseModel):
    type: str  # voiceover, music, both
    prompt: str

class GenerationRequest(BaseModel):
    prompt: str
    brandData: BrandData
    outputType: str  # static or reel
    frameCount: int
    audio: Optional[AudioConfig]

class GenerationResponse(BaseModel):
    imageUrls: List[str]
    audioUrl: Optional[str]
    status: str

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_brand_context(brand_data: BrandData) -> str:
    """Extract text context from brand data for prompt enhancement"""
    
    context_parts = []
    
    # Add color context
    colors = brand_data.colors
    context_parts.append(f"brand colors: primary {colors.primary}, secondary {colors.secondary}, accent {colors.accent}")
    
    # Add font context
    context_parts.append(f"using {brand_data.font} font style")
    
    # Add brand description if provided
    if brand_data.description:
        context_parts.append(f"brand personality: {brand_data.description}")
    
    # Style preferences from reference count
    if len(brand_data.references) > 0:
        context_parts.append(f"following provided brand style guidelines")
    
    return ", ".join(context_parts)

def enhance_prompt_with_brand(prompt: str, brand_data: BrandData) -> str:
    """Enhance user prompt with brand context"""
    
    brand_context = extract_brand_context(brand_data)
    
    enhanced = f"{prompt}, professional design, {brand_context}, high quality, clean layout, modern aesthetic"
    
    return enhanced

async def generate_image_replicate(prompt: str, seed: Optional[int] = None) -> str:
    """Generate image using Replicate's SDXL"""
    
    try:
        input_params = {
            "prompt": prompt,
            "negative_prompt": "low quality, blurry, distorted, ugly, bad composition, text, watermark",
            "width": 1024,
            "height": 1024,
            "num_outputs": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }
        
        if seed:
            input_params["seed"] = seed
        
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input=input_params
        )
        
        # Output is a list of URLs
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        
        return str(output)
        
    except Exception as e:
        print(f"Replicate generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

def apply_brand_overlay(image_url: str, brand_data: BrandData) -> str:
    """
    Download image, apply brand logo overlay, return new URL
    This is a placeholder - in production you'd upload to cloud storage
    """
    # For MVP, we'll return the original URL
    # TODO: Implement actual overlay with PIL
    return image_url

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Brand AI Generator API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_content(request: GenerationRequest):
    """
    Main generation endpoint
    Generates brand-consistent images based on prompt and brand data
    """
    
    try:
        print(f"Generating {request.outputType} content with {request.frameCount} frames")
        print(f"Prompt: {request.prompt}")
        
        # Enhance prompt with brand context
        enhanced_prompt = enhance_prompt_with_brand(request.prompt, request.brandData)
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        generated_urls = []
        
        # Generate images
        if request.outputType == "static":
            # Single image
            image_url = await generate_image_replicate(enhanced_prompt)
            # Apply brand overlay (logo, etc.)
            final_url = apply_brand_overlay(image_url, request.brandData)
            generated_urls.append(final_url)
            
        elif request.outputType == "reel":
            # Multiple frames with consistent seed base
            base_seed = 42  # You can randomize this
            
            for i in range(request.frameCount):
                # Vary the prompt slightly for each frame
                frame_prompt = f"{enhanced_prompt}, frame {i+1} of {request.frameCount}, dynamic composition"
                
                # Use different seeds for variety but keep base consistent
                seed = base_seed + i * 100
                
                image_url = await generate_image_replicate(frame_prompt, seed)
                final_url = apply_brand_overlay(image_url, request.brandData)
                generated_urls.append(final_url)
                
                print(f"Frame {i+1}/{request.frameCount} generated")
        
        # Audio generation (placeholder)
        audio_url = None
        if request.audio:
            # TODO: Integrate ElevenLabs or similar for audio
            print(f"Audio requested: {request.audio.type}")
            audio_url = None  # Placeholder
        
        return GenerationResponse(
            imageUrls=generated_urls,
            audioUrl=audio_url,
            status="success"
        )
        
    except Exception as e:
        print(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-brand")
async def process_brand(brand_data: BrandData):
    """
    Process and validate brand assets
    Extract brand embedding (placeholder for future ML model)
    """
    
    try:
        # Extract brand context
        context = extract_brand_context(brand_data)
        
        # In production, you would:
        # 1. Process logo with CLIP or similar
        # 2. Extract dominant colors from references
        # 3. Create brand embedding vector
        # 4. Store in vector database
        
        return {
            "status": "success",
            "message": "Brand assets processed",
            "context": context,
            "brandId": "temp_brand_id_123"  # Would be actual DB ID
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
