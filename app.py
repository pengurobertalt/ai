import os
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Set your Hugging Face token as environment variable HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("Set HF_TOKEN environment variable with your Hugging Face token")

model_id = "runwayml/stable-diffusion-v1-5"  # change to desired model e.g. stabilityai/stable-diffusion-2-1

def load_pipeline():
    # Use fp16 on GPU for speed if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=HF_TOKEN,
        torch_dtype=torch_dtype,
    )
    if device == "cuda":
        pipe = pipe.to(device)
    # Optional: enable xformers for speed (if compiled)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

pipe = load_pipeline()

def generate(prompt: str, guidance_scale: float = 7.5, num_inference_steps: int = 25, seed: int = None):
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
    image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=int(num_inference_steps), generator=generator).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=2, label="Prompt"),
        gr.Slider(1, 20, value=7.5, label="Guidance scale"),
        gr.Slider(1, 50, value=25, step=1, label="Inference steps"),
        gr.Number(label="Seed (optional)")
    ],
    outputs=gr.Image(type="pil"),
    title="Stable Diffusion demo",
    description="Minimal Stable Diffusion + Gradio demo. Make sure HF_TOKEN env var is set."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)