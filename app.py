# fluxdev_app.py
import os, torch
from uuid     import uuid4
from fastapi  import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import JSONResponse
from fastapi.staticfiles     import StaticFiles
from pydantic import BaseModel
from diffusers import FluxPipeline

API_KEY   = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID  = "black-forest-labs/FLUX.1-dev"
OUT_DIR   = os.path.join(os.path.dirname(__file__), "flux_images")
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.wildmindai.com",
        "https://api.wildmindai.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/fluxdev/images", StaticFiles(directory=OUT_DIR), name="flux-img")

print("🔄  Loading FLUX-Dev …")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅  FLUX-Dev ready!")

class Prompt(BaseModel):
    prompt: str
    height:   int = 512
    width:    int = 512
    steps:    int = 50
    guidance: float = 6.5
    seed:     int = 42

@app.post("/fluxdev")
async def generate(req: Request, body: Prompt):
    if req.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="bad api key")

    img = pipe(
        body.prompt.strip(),
        height=body.height,
        width=body.width,
        num_inference_steps=body.steps,
        guidance_scale=body.guidance,
        generator=torch.manual_seed(body.seed),
    ).images[0]

    fname = f"{uuid4().hex}.png"
    fpath = os.path.join(OUT_DIR, fname)
    img.save(fpath)
    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com/fluxdev/images/{fname}"}
    )
