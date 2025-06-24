from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
import torch
from diffusers import FluxPipeline
from uuid import uuid4
import os

app = FastAPI()

# Load model at startup
print("🔄 Loading FLUX Dev pipeline...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX Dev loaded!")

# Request schema
class PromptRequest(BaseModel):
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 50
    guidance: float = 6.5
    seed: int = 42

@app.post("/generate")
async def generate_image(data: PromptRequest):
    generator = torch.manual_seed(data.seed)
    image = pipe(
        prompt=data.prompt,
        height=data.height,
        width=data.width,
        guidance_scale=data.guidance,
        num_inference_steps=data.steps,
        generator=generator
    ).images[0]

    # Save with UUID
    image_path = f"output_{uuid4().hex}.png"
    image.save(image_path)
    return FileResponse(image_path, media_type="image/png", filename=image_path)
