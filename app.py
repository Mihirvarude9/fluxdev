"""
Flux Dev image-generation backend
--------------------------------
Runs on :  http://127.0.0.1:7863
Public:   https://api.wildmindai.com/flux
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from uuid import uuid4
import torch, os
from diffusers import DiffusionPipeline   # generic loader

# ---------- CONFIG ----------
API_KEY  = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID = "black-forest-labs/FLUX.1-dev"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- APP ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/flux/images", StaticFiles(directory=OUTPUT_DIR), name="flux-images")

# ---------- LOAD MODEL ----------
print("🔄 Loading FLUX Dev (this can take ~1-2 min)…")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",           # repo ships fp16 weights
    use_safetensors=True,
    trust_remote_code=True    # required for custom FluxPipeline
)
pipe.to("cuda")
pipe.enable_sequential_cpu_offload()      # extra VRAM savings
torch.cuda.empty_cache()
print("✅ FLUX Dev ready!")

# ---------- SCHEMA ----------
class PromptRequest(BaseModel):
    prompt: str
    height: int = 512
    width:  int = 512
    steps: int = 50
    guidance: float = 6.5
    seed: int = 42

# ---------- ROUTES ----------
@app.get("/flux/ping")
def ping():
    return {"status": "ok"}

@app.post("/flux")
async def generate_flux(request: Request, body: PromptRequest):
    # --- Auth ---------------------------------------------------
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # --- Generate ----------------------------------------------
    generator = torch.Generator(device="cuda").manual_seed(body.seed)
    result = pipe(
        prompt,
        height=body.height,
        width=body.width,
        guidance_scale=body.guidance,
        num_inference_steps=body.steps,
        generator=generator
    )
    image = result.images[0]

    # --- Save & respond ----------------------------------------
    filename = f"{uuid4().hex}.png"
    filepath  = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print(f"✅ Saved ⇒ {filepath}")

    return JSONResponse({
        "image_url": f"https://api.wildmindai.com/flux/images/{filename}"
    })
