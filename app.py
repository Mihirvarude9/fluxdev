from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
import torch
from diffusers import FluxPipeline
import os

# === CONFIG ===
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID = "black-forest-labs/FLUX.1-dev"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD FLUX MODEL ===
pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

# === FASTAPI APP SETUP ===
app = FastAPI()

# ✅ CORS Middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static images from /flux/images
app.mount("/flux/images", StaticFiles(directory=OUTPUT_DIR), name="flux-images")

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str

# === POST /flux Endpoint ===
@app.post("/flux")
async def generate_flux(request: Request, body: PromptRequest):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # 🔥 Generate image
    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=6.5,
        num_inference_steps=50,
        generator=torch.manual_seed(42)
    ).images[0]

    # Save and return image
    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {
        "image_url": f"https://api.wildmindai.com/flux/images/{filename}"
    }
