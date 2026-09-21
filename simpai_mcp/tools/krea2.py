"""Krea2 tools: txt2img, image edit, LoRA support, and batch generation."""
from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from ..client import ComfyClient

# --- Output dir: follow SimpAI convention at users/Local/outputs-mcp ---
OUTPUT_DIR = Path(r"I:\SimpAI\users\Local\outputs-mcp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR = OUTPUT_DIR / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
BATCH_REGISTRY = OUTPUT_DIR / "batch_registry.json"


def _timestamp_name() -> str:
    """Generate base filename like 2026-09-19_01-50-52_3639."""
    return time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(random.randint(1000, 9999))

# --- Krea2 model names (verified from frontend log.html) ---
KREA2_UNET = r"Krea2\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors"
KREA2_CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
KREA2_VAE = "qwen_image_vae.safetensors"


def _load_registry() -> dict:
    if BATCH_REGISTRY.exists():
        return json.loads(BATCH_REGISTRY.read_text(encoding="utf-8"))
    return {}


def _save_registry(reg: dict) -> None:
    BATCH_REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_loras(
    wf: dict,
    loras: list[dict],
    model_node: str = "1",
    clip_node: str = "2",
) -> None:
    """Insert LoraLoader nodes between model/clip and downstream.

    loras: [{"name": "xxx.safetensors", "strength": 0.8}, ...]
    """
    if not loras:
        return
    prev_model = [model_node, 0]
    prev_clip = [clip_node, 0]
    for i, lora in enumerate(loras):
        node_id = f"lora_{i}"
        wf[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": lora.get("strength", 1.0),
                "strength_clip": lora.get("strength_clip", lora.get("strength", 1.0)),
                "model": prev_model,
                "clip": prev_clip,
            },
        }
        prev_model = [node_id, 0]
        prev_clip = [node_id, 1]
    # Redirect KSampler.model and CLIPTextEncode.clip to last lora
    wf["7"]["inputs"]["model"] = prev_model
    wf["4"]["inputs"]["clip"] = prev_clip
    wf["5"]["inputs"]["clip"] = prev_clip


def _build_txt2img_workflow(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    loras: list[dict] | None = None,
) -> dict:
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA2_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA2_CLIP, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "simpai_mcp"}},
    }
    _apply_loras(wf, loras or [])
    return wf


def _build_edit_workflow(
    prompt: str,
    negative_prompt: str,
    image_filename: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    denoise: float = 0.75,
    loras: list[dict] | None = None,
) -> dict:
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KREA2_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KREA2_CLIP, "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KREA2_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["2", 0]}},
        "10": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "euler", "scheduler": "simple", "denoise": denoise,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["11", 0],
        }},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "simpai_edit"}},
    }
    _apply_loras(wf, loras or [])
    return wf


# ---------- single image tools ----------

async def krea2_txt2img(
    client: ComfyClient,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
    loras: list[dict] | None = None,
) -> dict:
    used_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    wf = _build_txt2img_workflow(prompt, negative_prompt, width, height, steps, cfg, used_seed, loras)
    prompt_id = await client.submit_prompt(wf)
    return {"prompt_id": prompt_id, "seed": used_seed, "mode": "txt2img", "lora_count": len(loras or [])}


async def krea2_edit_image(
    client: ComfyClient,
    prompt: str,
    image_path: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
    denoise: float = 0.75,
    loras: list[dict] | None = None,
) -> dict:
    filename = await client.upload_image(image_path)
    used_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    wf = _build_edit_workflow(prompt, negative_prompt, filename, width, height, steps, cfg, used_seed, denoise, loras)
    prompt_id = await client.submit_prompt(wf)
    return {"prompt_id": prompt_id, "seed": used_seed, "uploaded_image": filename, "mode": "edit", "denoise": denoise}


# ---------- result polling ----------

async def krea2_get_result(
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
            return {
                "prompt_id": prompt_id,
                "done": True,
                "status": status.get("status_str", "success"),
                "images": saved,
            }
        if not wait or time.time() > deadline:
            q = await client.queue()
            return {
                "prompt_id": prompt_id,
                "done": False,
                "running": len(q.get("queue_running", [])),
                "pending": len(q.get("queue_pending", [])),
            }
        await asyncio.sleep(poll_interval)


# ---------- batch tools ----------

async def krea2_batch_generate(
    client: ComfyClient,
    shots: list[dict],
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed_base: int | None = None,
    loras: list[dict] | None = None,
) -> dict:
    """Submit a batch of text-to-image shots.

    shots: [{"shot_id": "01", "prompt": "...", "negative_prompt": "..."}, ...]
    Returns batch_id and per-shot prompt_ids.
    """
    batch_id = f"batch_{int(time.time())}"
    reg = _load_registry()
    entries: dict[str, dict] = {}
    base = seed_base if seed_base is not None else random.randint(1, 2**30)

    for i, shot in enumerate(shots):
        shot_id = str(shot.get("shot_id", f"{i+1:02d}"))
        prompt = shot["prompt"]
        neg = shot.get("negative_prompt", "")
        seed = base + i
        wf = _build_txt2img_workflow(prompt, neg, width, height, steps, cfg, seed, loras)
        prompt_id = await client.submit_prompt(wf)
        entries[shot_id] = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "seed": seed,
            "done": False,
            "images": [],
        }
        await asyncio.sleep(0.5)  # small gap to avoid overwhelming the queue

    reg[batch_id] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(shots),
        "entries": entries,
    }
    _save_registry(reg)
    return {"batch_id": batch_id, "total": len(shots), "entries": {k: {"prompt_id": v["prompt_id"], "prompt": v["prompt"]} for k, v in entries.items()}}


async def krea2_batch_status(client: ComfyClient, batch_id: str) -> dict:
    reg = _load_registry()
    if batch_id not in reg:
        return {"error": f"batch {batch_id} not found"}
    batch = reg[batch_id]
    done = 0
    pending = 0
    for shot_id, entry in batch["entries"].items():
        if entry["done"]:
            done += 1
            continue
        history = await client.get_history(entry["prompt_id"])
        if history:
            images = await client.list_output_images(entry["prompt_id"])
            saved = []
            for i, img in enumerate(images):
                data = await client.view_image(img["filename"], img["subfolder"], img["type"])
                suffix = "" if i == 0 else f"_{i}"
                out_path = BATCH_DIR / batch_id / f"{shot_id}_{_timestamp_name()}{suffix}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(data)
                saved.append(str(out_path))
            entry["done"] = True
            entry["images"] = saved
            done += 1
        else:
            pending += 1
    _save_registry(reg)
    return {
        "batch_id": batch_id,
        "total": batch["total"],
        "done": done,
        "pending": pending,
        "all_done": done == batch["total"],
        "entries": {k: {"done": v["done"], "images": v["images"]} for k, v in batch["entries"].items()},
    }


async def krea2_batch_results(client: ComfyClient, batch_id: str) -> dict:
    reg = _load_registry()
    if batch_id not in reg:
        return {"error": f"batch {batch_id} not found"}
    batch = reg[batch_id]
    results = {}
    for shot_id, entry in batch["entries"].items():
        if not entry["done"]:
            history = await client.get_history(entry["prompt_id"])
            if history:
                images = await client.list_output_images(entry["prompt_id"])
                saved = []
                for i, img in enumerate(images):
                    data = await client.view_image(img["filename"], img["subfolder"], img["type"])
                    suffix = "" if i == 0 else f"_{i}"
                    out_path = BATCH_DIR / batch_id / f"{shot_id}_{_timestamp_name()}{suffix}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(data)
                    saved.append(str(out_path))
                entry["done"] = True
                entry["images"] = saved
        results[shot_id] = entry["images"]
    _save_registry(reg)
    return {"batch_id": batch_id, "results": results}
