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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/fluxdev/images", StaticFiles(directory=OUTPUT_DIR), name="fluxdev-images")

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

@app.options("/fluxdev")                 # handle the pre-flight
def fluxdev_options():
    return JSONResponse(content=None, status_code=204)

@app.post("/fluxdev")
async def generate_fluxdev(request: Request, body: PromptRequest):
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")

    img = pipe(
        body.prompt.strip(),
        height=body.height,
        width=body.width,
        num_inference_steps=body.steps,
        guidance_scale=body.guidance,
        generator=torch.manual_seed(body.seed)
    ).images[0]

    fname = f"{uuid4().hex}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    img.save(fpath)

    return {
        "image_url": f"https://api.wildmindai.com/fluxdev/images/{fname}"
    }
