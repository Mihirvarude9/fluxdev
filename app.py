import os, torch
from uuid import uuid4
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from diffusers import FluxPipeline

# ───────── CONFIG ─────────────────────────────────────────────
API_KEY  = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
PREFIX   = "/fluxdev"                       # <── all paths in one place

BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_fluxdev")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ───────── APP ────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.wildmindai.com",
        "https://api.wildmindai.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(f"{PREFIX}/images", StaticFiles(directory=OUTPUT_DIR),
          name="fluxdev-images")

# ───────── MODEL ──────────────────────────────────────────────
print("🔄 Loading FLUX-Dev …")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX-Dev ready!")

# ───────── SCHEMA ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str
    height: int = 512
    width:  int = 512
    steps:  int = 50
    guidance: float = 6.5
    seed: int = 42

# ───────── ROUTES ─────────────────────────────────────────────
@app.get(f"{PREFIX}/ping")
def ping():
    return {"status": "ok"}

@app.post(PREFIX)
async def generate(request: Request, body: PromptRequest):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    image = pipe(
        prompt,
        height=body.height,
        width=body.width,
        num_inference_steps=body.steps,
        guidance_scale=body.guidance,
        generator=torch.manual_seed(body.seed)
    ).images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print("🖼️  saved", filepath)

    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com{PREFIX}/images/{filename}"}
    )
