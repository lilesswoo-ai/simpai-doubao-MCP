"""MiniMax-H3 R2I: text/reference-to-image generation."""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from ..client import ComfyClient
from .krea2 import OUTPUT_DIR, _timestamp_name

# --- MiniMax-H3 model names ---
H3_UNET = "minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
H3_TURBO_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"


def _build_h3_r2i(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    cfg: float = 1.0,
    seed: int = 0,
    ref_image: str | None = None,
) -> dict:
    """Build MiniMax-H3 R2I workflow using standard + SimpAI custom nodes."""
    wf = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": H3_CLIP, "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE}},
        "2": {"class_type": "SeedInput", "inputs": {"seed": seed}},

        # Adaptive reference: encodes prompt + ref image, outputs [positive, latent]
        "11": {"class_type": "SimpAIMiniMaxH3AdaptiveReference", "inputs": {
            "clip": ["4", 0],
            "vae": ["5", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": 124,
            "ref_image_size": "auto",
            "reference_token_budget": 0,
            "max_image_long_edge": 2048,
        }},

        # Turbo LoRA
        "59": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "lora_name": H3_TURBO_LORA,
            "strength_model": 1.0,
            "model": ["3", 0],
        }},

        # Sigma shift
        "69": {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": ["59", 0],
            "shift_video": 12.0,
            "shift_audio": 3.0,
        }},

        # Sampler chain
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": ["69", 0],
            "scheduler": "beta57",
            "steps": steps,
            "denoise": 1.0,
        }},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": ["2", 0]}},
        "12": {"class_type": "BasicGuider", "inputs": {
            "model": ["69", 0],
            "conditioning": ["11", 0],
        }},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0],
            "guider": ["12", 0],
            "sampler": ["8", 0],
            "sigmas": ["9", 0],
            "latent_image": ["11", 1],
        }},

        # Decode
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["5", 0]}},
        "15": {"class_type": "ImageFromBatch", "inputs": {"image": ["14", 0], "batch_index": 0, "length": 1}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["15", 0], "filename_prefix": "h3_mcp"}},
    }

    # Add reference image if provided
    if ref_image:
        wf["18"] = {"class_type": "LoadImage", "inputs": {"image": ref_image}}
        wf["11"]["inputs"]["ref_images"] = ["18", 0]

    return wf


async def h3_r2i_generate(
    client: ComfyClient,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    cfg: float = 1.0,
    seed: int | None = None,
    ref_image_path: str | None = None,
) -> dict:
    used_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    ref_name = None
    if ref_image_path:
        ref_name = await client.upload_image(ref_image_path)
    wf = _build_h3_r2i(prompt, width, height, steps, cfg, used_seed, ref_name)
    prompt_id = await client.submit_prompt(wf)
    return {
        "prompt_id": prompt_id,
        "seed": used_seed,
        "mode": "minimax_h3_r2i",
        "ref_image": ref_name,
    }


async def h3_get_result(
    client: ComfyClient,
    prompt_id: str,
    wait: bool = False,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
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
