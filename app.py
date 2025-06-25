# fluxdev_app.py   (run with:  uvicorn fluxdev_app:app --host 0.0.0.0 --port 7865)
import os
from uuid import uuid4
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import FluxPipeline

API_KEY   = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID  = "black-forest-labs/FLUX.1-dev"
BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# CORS middleware - more explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com", "https://api.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key", "Accept"],
)

# Mount static files FIRST with a different path to avoid conflicts
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="flux-images")

print("🔄 Loading FLUX-Dev …")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX-Dev ready!")

class PromptRequest(BaseModel):
    prompt:   str
    height:   int = 512
    width:    int = 512
    steps:    int = 50
    guidance: float = 6.5
    seed:     int = 42

# Explicit OPTIONS handler (though CORS middleware should handle this)
@app.options("/fluxdev")
async def fluxdev_options():
    return JSONResponse(
        content=None, 
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "https://www.wildmindai.com",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, x-api-key, Accept",
        }
    )

@app.post("/fluxdev")
async def generate_fluxdev(request: Request, body: PromptRequest):
    # Validate API key
    api_key = request.headers.get("x-api-key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Generate image
        img = pipe(
            body.prompt.strip(),
            height=body.height,
            width=body.width,
            num_inference_steps=body.steps,
            guidance_scale=body.guidance,
            generator=torch.manual_seed(body.seed)
        ).images[0]
        
        # Save image
        fname = f"{uuid4().hex}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)
        img.save(fpath)
        
        # Return response with updated image URL path
        return JSONResponse(
            content={
                "image_url": f"https://api.wildmindai.com/images/{fname}"
            },
            headers={
                "Access-Control-Allow-Origin": "https://www.wildmindai.com",
            }
        )
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "FLUX.1-dev"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "FLUX.1-dev API Server", "endpoints": ["/fluxdev", "/health"]}