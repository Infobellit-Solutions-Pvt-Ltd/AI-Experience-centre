from diffusers import DiffusionPipeline
import torch

# Load the model once at startup on the GPU
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_gpu=True)
print("Loading complete")
