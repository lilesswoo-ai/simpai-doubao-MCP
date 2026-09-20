"""ComfyUI (comfyd) HTTP client.

Talks only to the ComfyUI HTTP API (default http://127.0.0.1:8188).
Never imports SimpAI source.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import httpx


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    # ---------- basic ----------
    async def system_stats(self) -> dict[str, Any]:
        r = await self._http.get("/api/system_stats")
        r.raise_for_status()
        return r.json()

    async def queue(self) -> dict[str, Any]:
        r = await self._http.get("/api/queue")
        r.raise_for_status()
        return r.json()

    async def interrupt(self) -> dict[str, Any]:
        r = await self._http.post("/api/interrupt")
        r.raise_for_status()
        return r.json()

    # ---------- images ----------
    async def upload_image(self, local_path: str | Path, subfolder: str = "") -> str:
        """Upload a local image into ComfyUI input dir. Returns the filename used in LoadImage."""
        p = Path(local_path)
        with p.open("rb") as f:
            files = {"image": (p.name, f, "image/png")}
            data = {"overwrite": "true", "type": "input", "subfolder": subfolder}
            r = await self._http.post("/api/upload/image", files=files, data=data)
        r.raise_for_status()
        body = r.json()
        # ComfyUI returns {"name": "...", "subfolder": "...", "type": "input"}
        return body.get("name", p.name)

    async def view_image(self, filename: str, subfolder: str = "", img_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
        r = await self._http.get("/api/view", params=params)
        r.raise_for_status()
        return r.content

    # ---------- prompt / history ----------
    async def submit_prompt(self, workflow: dict[str, Any]) -> str:
        """Submit an API-format workflow. Returns prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        r = await self._http.post("/api/prompt", json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"submit_prompt failed: {r.status_code} {r.text}")
        body = r.json()
        return body["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        r = await self._http.get(f"/api/history/{prompt_id}")
        r.raise_for_status()
        data = r.json()
        return data.get(prompt_id, {})

    async def is_done(self, prompt_id: str) -> bool:
        h = await self.get_history(prompt_id)
        return bool(h)

    async def list_output_images(self, prompt_id: str) -> list[dict[str, str]]:
        """Extract output image descriptors from history."""
        h = await self.get_history(prompt_id)
        images: list[dict[str, str]] = []
        outputs = h.get("outputs", {})
        for _node_id, out in outputs.items():
            for img in out.get("images", []):
                images.append(
                    {
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    }
                )
        # SaveImageWebsocketLazy may also surface via gifs/previews
        return images
