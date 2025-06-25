"""
Run with:
    uvicorn flux_app:app --host 0.0.0.0 --port 7863
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import torch, os
from diffusers import FluxPipeline  # keep this import after torch to avoid warnings

# ---------- CONFIG ----------
API_KEY   = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID  = "black-forest-labs/FLUX.1-dev"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- APP ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],  # <- your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images at /flux/images/...
app.mount("/flux/images", StaticFiles(directory=OUTPUT_DIR),
          name="flux-images")

# ---------- LOAD MODEL ----------
print("🔄 Loading FLUX Dev...")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX Dev ready!")

# ---------- SCHEMAS ----------
class PromptRequest(BaseModel):
    prompt: str
    height: int = 512
    width:  int = 512
    steps:  int = 50
    guidance: float = 6.5
    seed: int = 42

# ---------- ROUTES ----------
@app.get("/flux/ping")
def ping():                     # tiny health-check
    return {"status": "ok"}

@app.post("/flux")
async def generate_flux(request: Request, body: PromptRequest):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # --- Generate image ---
    image = pipe(
        prompt,
        height=body.height,
        width=body.width,
        guidance_scale=body.guidance,
        num_inference_steps=body.steps,
        generator=torch.manual_seed(body.seed)
    ).images[0]

    # --- Save & respond ---
    filename = f"{uuid4().hex}.png"
    filepath  = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print(f"✅ Saved: {filepath}")

    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com/flux/images/{filename}"}
    )
