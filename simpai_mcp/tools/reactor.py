"""ReActor Face Swap: swap faces in images using ReActorFaceSwap node."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..client import ComfyClient
from .krea2 import OUTPUT_DIR, _timestamp_name


def _build_reactor_swap(
    target_image: str,
    source_image: str,
    input_face_index: str = "0",
    source_face_index: str = "0",
    face_restore_model: str = "codeformer-v0.1.0.pth",
    face_restore_visibility: float = 1.0,
    codeformer_weight: float = 0.5,
) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": target_image}},
        "2": {"class_type": "LoadImage", "inputs": {"image": source_image}},
        "3": {"class_type": "ReActorFaceSwap", "inputs": {
            "enabled": True,
            "input_image": ["1", 0],
            "source_image": ["2", 0],
            "swap_model": "inswapper_128.onnx",
            "facedetection": "retinaface_resnet50",
            "face_restore_model": face_restore_model,
            "face_restore_visibility": face_restore_visibility,
            "codeformer_weight": codeformer_weight,
            "detect_gender_input": "no",
            "detect_gender_source": "no",
            "input_faces_index": input_face_index,
            "source_faces_index": source_face_index,
            "console_log_level": 1,
        }},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "reactor_mcp"}},
    }


async def reactor_swap_face(
    client: ComfyClient,
    target_image_path: str,
    source_face_path: str,
    input_face_index: str = "0",
    source_face_index: str = "0",
    face_restore_model: str = "codeformer-v0.1.0.pth",
    face_restore_visibility: float = 1.0,
    codeformer_weight: float = 0.5,
) -> dict:
    target = await client.upload_image(target_image_path)
    source = await client.upload_image(source_face_path)
    wf = _build_reactor_swap(
        target, source,
        input_face_index=input_face_index,
        source_face_index=source_face_index,
        face_restore_model=face_restore_model,
        face_restore_visibility=face_restore_visibility,
        codeformer_weight=codeformer_weight,
    )
    prompt_id = await client.submit_prompt(wf)
    return {
        "prompt_id": prompt_id,
        "mode": "reactor_faceswap",
        "target_image": target,
        "source_face": source,
    }


async def reactor_get_result(
    client: ComfyClient,
    prompt_id: str,
    wait: bool = False,
    timeout: float = 120.0,
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
