from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import FluxPipeline
import torch
import os
from uuid import uuid4

# === CONFIG ===
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD FLUX MODEL ===
print("🔄 Loading FLUX Dev model...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX Dev model ready!")

# === FASTAPI SETUP ===
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# === Input schema ===
class PromptRequest(BaseModel):
    prompt: str

# === /flux endpoint ===
@app.post("/flux")
async def generate_flux(request: Request, body: PromptRequest):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    image = pipe(
        prompt=prompt,
        height=512,
        width=512,
        guidance_scale=6.5,
        num_inference_steps=50,
        generator=torch.manual_seed(42)
    ).images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"image_url": f"https://api.wildmindai.com/images/{filename}"}
