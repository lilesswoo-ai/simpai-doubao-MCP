"""Qwen Image 2.1 - text-to-image and image-edit workflow."""
import asyncio, sys, os
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient
from pathlib import Path
from datetime import datetime
import random

# === Qwen Image 2512 config (verified working) ===
T2I_UNET = "qwen_image_2512_fp8_e4m3fn.safetensors"
EDIT_UNET = "qwen_image_edit_2511_fp8mixed.safetensors"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
SAMPLER = "euler"
SCHEDULER = "beta"
STEPS = 8
CFG = 1.0

OUT_DIR = Path(r"I:\SimpAI\users\Local\outputs-mcp")

def make_t2i_workflow(prompt, neg, width=1280, height=720, seed=None):
    if seed is None:
        seed = random.randint(1, 999999)
    return {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": T2I_UNET, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": neg}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["3", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": seed,
            "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "qwen_t2i"}},
    }, seed

def make_edit_workflow(prompt, input_image_name, width=1280, height=720, seed=None):
    """Image edit using TextEncodeQwenImageEditPlusPro_lrzjason custom node."""
    if seed is None:
        seed = random.randint(1, 999999)
    return {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": EDIT_UNET, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "qwen_image"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "100": {"class_type": "LoadImage", "inputs": {"image": input_image_name}},
        "101": {"class_type": "TextEncodeQwenImageEditPlusPro_lrzjason", "inputs": {
            "clip": ["4", 0], "vae": ["5", 0],
            "image1": ["100", 0],
            "prompt": prompt
        }},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["3", 0], "positive": ["101", 0], "negative": ["101", 0],
            "latent_image": ["101", 1], "seed": seed,
            "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "qwen_edit"}},
    }, seed

async def run_workflow(client, wf, label=""):
    pid = await client.submit_prompt(wf)
    for _ in range(300):
        await asyncio.sleep(1)
        if await client.is_done(pid):
            break
    imgs = await client.list_output_images(pid)
    if not imgs:
        print(f"[{label}] No output")
        return None
    img = imgs[0]
    data = await client.view_image(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rand = f"{random.randint(1000,9999)}"
    out = OUT_DIR / f"qwen_{label}_{ts}_{rand}.png"
    out.write_bytes(data)
    print(f"[{label}] Saved: {out}")
    return out

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        # Test 1: T2I
        prompt = ("Chinese giant aesthetics, 16:9, photorealistic. A beautiful woman in flowing white hanfu "
                  "standing on a giant lotus leaf floating on a misty lake at dawn. Golden sunrise, 8K.")
        neg = ""
        wf, seed = make_t2i_workflow(prompt, neg, width=1280, height=720)
        await run_workflow(client, wf, f"t2i_seed{seed}")
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
