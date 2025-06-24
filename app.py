from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
from diffusers import StableDiffusionPipeline  # or your specific flux pipeline class
import torch
import os

# === CONFIG ===
model_id = "your-flux-model-id-or-path"  # Replace this with your actual model path
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD MODEL ===
# Replace this with your actual Flux-compatible pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# === FASTAPI SETUP ===
app = FastAPI()

# ✅ Enable CORS for frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static images under /flux/images/
app.mount("/flux/images", StaticFiles(directory=OUTPUT_DIR), name="flux-images")

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str

# === Flux Generation Endpoint ===
@app.post("/flux")
async def generate_flux(request: Request, body: PromptRequest):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # 🔥 Generate image
    image = pipeline(
        prompt=prompt,
        num_inference_steps=40,
        guidance_scale=5.5
    ).images[0]

    # Save image
    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print(f"✅ Image saved at: {filepath}")

    # Return public URL
    return {"image_url": f"https://api.wildmindai.com/flux/images/{filename}"}
