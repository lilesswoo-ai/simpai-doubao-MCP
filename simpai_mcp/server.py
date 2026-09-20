"""SimpAI MCP Server entry point.

Run with:
    python -m simpai_mcp.server   (stdio transport)
"""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .client import ComfyClient
from .tools import krea2, zimage, reactor, minimax_h3

BASE_URL = os.environ.get("SIMPAI_COMFYD_URL", "http://127.0.0.1:8188")

mcp = FastMCP("simpai-mcp")
_client: ComfyClient | None = None


def _get_client() -> ComfyClient:
    global _client
    if _client is None:
        _client = ComfyClient(BASE_URL)
    return _client


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def krea2_txt2img(
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
    loras: list[dict] | None = None,
) -> dict:
    """Krea2 text-to-image. Submit a generation job and return a prompt_id.

    Args:
        prompt: Positive prompt describing the image.
        negative_prompt: Negative prompt (optional).
        width: Output width (default 832).
        height: Output height (default 1216).
        steps: Sampling steps (default 8).
        cfg: CFG scale (default 1.0).
        seed: Random seed; omit for random.
        loras: Optional LoRA list, e.g. [{"name": "xxx.safetensors", "strength": 0.8}].
    """
    return await krea2.krea2_txt2img(
        _get_client(), prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, cfg=cfg, seed=seed, loras=loras,
    )


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def krea2_edit_image(
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
    """Krea2 image edit. Upload a local image and submit an edit job.

    Args:
        prompt: Edit instruction.
        image_path: Absolute local path of the input image.
        negative_prompt: Negative prompt (optional).
        width: Output width (default 1024).
        height: Output height (default 1024).
        steps: Sampling steps (default 8).
        cfg: CFG scale (default 1.0).
        seed: Random seed; omit for random.
        denoise: How much to change (0.3=subtle, 0.75=moderate, 1.0=full, default 0.75).
        loras: Optional LoRA list.
    """
    return await krea2.krea2_edit_image(
        _get_client(), prompt=prompt, image_path=image_path,
        negative_prompt=negative_prompt, width=width, height=height,
        steps=steps, cfg=cfg, seed=seed, denoise=denoise, loras=loras,
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def krea2_get_result(
    prompt_id: str,
    wait: bool = False,
    timeout: float = 180.0,
) -> dict:
    """Poll a Krea2 job and download output images when done.

    Args:
        prompt_id: The id returned by krea2_txt2img or krea2_edit_image.
        wait: If true, block until done or timeout.
        timeout: Max seconds to wait (default 180).
    """
    return await krea2.krea2_get_result(_get_client(), prompt_id, wait=wait, timeout=timeout)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def krea2_batch_generate(
    shots: list[dict],
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed_base: int | None = None,
    loras: list[dict] | None = None,
) -> dict:
    """Submit a batch of text-to-image shots (e.g. storyboard frames).

    Args:
        shots: List of shot dicts, e.g.
            [{"shot_id": "01", "prompt": "...", "negative_prompt": "..."}, ...]
        width: Output width (default 832).
        height: Output height (default 1216).
        steps: Sampling steps (default 8).
        cfg: CFG scale (default 1.0).
        seed_base: Base seed; each shot gets seed_base + index. Omit for random.
        loras: Optional LoRA list applied to all shots.
    """
    return await krea2.krea2_batch_generate(
        _get_client(), shots=shots, width=width, height=height,
        steps=steps, cfg=cfg, seed_base=seed_base, loras=loras,
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def krea2_batch_status(batch_id: str) -> dict:
    """Check batch progress and download completed shots.

    Args:
        batch_id: The id returned by krea2_batch_generate.
    """
    return await krea2.krea2_batch_status(_get_client(), batch_id)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def krea2_batch_results(batch_id: str) -> dict:
    """Fetch all batch results (waits for pending shots to finish).

    Args:
        batch_id: The id returned by krea2_batch_generate.
    """
    return await krea2.krea2_batch_results(_get_client(), batch_id)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def zimage_txt2img(
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Z-imageT Turbo text-to-image. Fast image generation with Z-imageT model.

    Args:
        prompt: Positive prompt.
        negative_prompt: Negative prompt (optional).
        width: Output width (default 832).
        height: Output height (default 1216).
        steps: Sampling steps (default 8).
        cfg: CFG scale (default 1.0).
        seed: Random seed; omit for random.
    """
    return await zimage.zimage_txt2img(
        _get_client(), prompt=prompt, negative_prompt=negative_prompt,
        width=width, height=height, steps=steps, cfg=cfg, seed=seed,
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def zimage_get_result(prompt_id: str, wait: bool = False, timeout: float = 180.0) -> dict:
    """Poll a Z-imageT job and download output images.

    Args:
        prompt_id: The id returned by zimage_txt2img.
        wait: If true, block until done or timeout.
        timeout: Max seconds to wait (default 180).
    """
    return await zimage.zimage_get_result(_get_client(), prompt_id, wait=wait, timeout=timeout)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def reactor_swap_face(
    target_image_path: str,
    source_face_path: str,
    input_face_index: str = "0",
    source_face_index: str = "0",
    face_restore_model: str = "codeformer-v0.1.0.pth",
    face_restore_visibility: float = 1.0,
    codeformer_weight: float = 0.5,
) -> dict:
    """ReActor face swap. Swap the face in target image with the face from source image.

    Args:
        target_image_path: Absolute local path of the target image (where the face goes).
        source_face_path: Absolute local path of the source face image.
        input_face_index: Which face index in target image (default "0").
        source_face_index: Which face index in source image (default "0").
        face_restore_model: Restore model (codeformer-v0.1.0.pth or GFPGANv1.4.pth).
        face_restore_visibility: Restore strength 0.1-1.0 (default 1.0).
        codeformer_weight: CodeFormer weight 0.0-1.0 (default 0.5).
    """
    return await reactor.reactor_swap_face(
        _get_client(), target_image_path=target_image_path, source_face_path=source_face_path,
        input_face_index=input_face_index, source_face_index=source_face_index,
        face_restore_model=face_restore_model, face_restore_visibility=face_restore_visibility,
        codeformer_weight=codeformer_weight,
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def reactor_get_result(prompt_id: str, wait: bool = False, timeout: float = 120.0) -> dict:
    """Poll a ReActor swap job and download the result.

    Args:
        prompt_id: The id returned by reactor_swap_face.
        wait: If true, block until done or timeout.
        timeout: Max seconds to wait (default 120).
    """
    return await reactor.reactor_get_result(_get_client(), prompt_id, wait=wait, timeout=timeout)


@mcp.tool(annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False})
async def h3_r2i_generate(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 10,
    cfg: float = 1.0,
    seed: int | None = None,
    ref_image_path: str | None = None,
) -> dict:
    """MiniMax-H3 R2I: text-to-image or reference-to-image generation.
    Supports up to 9 reference images. Large model (35GB), may take several minutes to load.

    Args:
        prompt: Positive prompt describing the image.
        width: Output width (default 1024).
        height: Output height (default 1024).
        steps: Sampling steps (default 10).
        cfg: CFG scale (default 1.0).
        seed: Random seed; omit for random.
        ref_image_path: Optional local path of a reference image for image editing.
    """
    return await minimax_h3.h3_r2i_generate(
        _get_client(), prompt=prompt, width=width, height=height,
        steps=steps, cfg=cfg, seed=seed, ref_image_path=ref_image_path,
    )


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
async def h3_get_result(prompt_id: str, wait: bool = False, timeout: float = 600.0) -> dict:
    """Poll a MiniMax-H3 R2I job and download output images.

    Args:
        prompt_id: The id returned by h3_r2i_generate.
        wait: If true, block until done or timeout.
        timeout: Max seconds to wait (default 600, H3 model is large).
    """
    return await minimax_h3.h3_get_result(_get_client(), prompt_id, wait=wait, timeout=timeout)


if __name__ == "__main__":
    mcp.run()
