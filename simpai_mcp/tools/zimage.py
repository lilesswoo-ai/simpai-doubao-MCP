"""Z-imageT Turbo: text-to-image using standard ComfyUI nodes."""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from ..client import ComfyClient
from .krea2 import OUTPUT_DIR, _timestamp_name

# --- Z-imageT model names (from preset) ---
ZIMAGE_UNET = "z_image_turbo_int8_convrot.safetensors"
ZIMAGE_CLIP = "qwen_3_4b.safetensors"
ZIMAGE_VAE = "ae.safetensors"


def _build_zimage_txt2img(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ZIMAGE_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ZIMAGE_CLIP, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ZIMAGE_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler_ancestral", "scheduler": "beta", "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "zimage_mcp"}},
    }


async def zimage_txt2img(
    client: ComfyClient,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
) -> dict:
    used_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    wf = _build_zimage_txt2img(prompt, negative_prompt, width, height, steps, cfg, used_seed)
    prompt_id = await client.submit_prompt(wf)
    return {"prompt_id": prompt_id, "seed": used_seed, "mode": "zimage_txt2img"}


async def zimage_get_result(
    client: ComfyClient,
    prompt_id: str,
    wait: bool = False,
    timeout: float = 180.0,
    poll_interval: float = 3.0,
) -> dict:
    deadline = time.time() + timeout if wait else 0.0
    while True:
        history = await client.get_history(prompt_id)
        if history:
            images = await client.list_output_images(prompt_id)
            saved: list[str] = []
            for i, img in enumerate(images):
                data = await client.view_image(img["filename"], img["subfolder"], img["type"])
                suffix = "" if i == 0 else f"_{i}"
                out_path = OUTPUT_DIR / f"{_timestamp_name()}{suffix}.png"
                out_path.write_bytes(data)
                saved.append(str(out_path))
            status = history.get("status", {})
            return {"prompt_id": prompt_id, "done": True, "status": status.get("status_str", "success"), "images": saved}
        if not wait or time.time() > deadline:
            q = await client.queue()
            return {"prompt_id": prompt_id, "done": False, "running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", []))}
        await asyncio.sleep(poll_interval)
