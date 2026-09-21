"""Template loader/filler.

Reads API-format workflow snapshots from ./snapshots (copied from SimpAI,
never mutated). Fills only the input nodes MCP knows about; unknown fields
keep template defaults so SimpAI updates that add fields do not break us.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def load_template(name: str) -> dict[str, Any]:
    """Load a workflow JSON snapshot by file stem, e.g. 'krea2_aio_cn_api'."""
    p = SNAPSHOTS_DIR / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"template not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _find_node_by_class(workflow: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]] | None:
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    return None


def _set_input(node: dict[str, Any], key: str, value: Any) -> bool:
    """Set a scalar input. Returns False if the input does not exist (kept default)."""
    inputs = node.setdefault("inputs", {})
    if key not in inputs:
        return False
    # Never overwrite a link (list like ["12", 0]) with a scalar.
    if isinstance(inputs[key], list):
        return False
    inputs[key] = value
    return True


def fill_general_input(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 832,
    height: int = 1216,
    steps: int = 8,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
) -> None:
    """Fill the AIO GeneralInput node (krea2_aio_cn_api uses node 27)."""
    found = _find_node_by_class(workflow, "GeneralInput")
    if found is None:
        raise ValueError("GeneralInput node not found in template")
    _, node = found
    _set_input(node, "prompt", prompt)
    if negative_prompt:
        _set_input(node, "negative_prompt", negative_prompt)
    _set_input(node, "width", width)
    _set_input(node, "height", height)
    _set_input(node, "steps", steps)
    _set_input(node, "cfg", cfg)
    _set_input(node, "sampler", sampler)
    _set_input(node, "scheduler", scheduler)


def fill_scene_input(
    workflow: dict[str, Any],
    *,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
) -> None:
    """Fill the SceneInput node (scene_krea2_edit_cn_api uses node 2)."""
    found = _find_node_by_class(workflow, "SceneInput")
    if found is None:
        raise ValueError("SceneInput node not found in template")
    _, node = found
    _set_input(node, "prompt", prompt)
    if negative_prompt:
        _set_input(node, "negative_prompt", negative_prompt)
    _set_input(node, "width", width)
    _set_input(node, "height", height)
    _set_input(node, "steps", steps)
    _set_input(node, "cfg", cfg)
    _set_input(node, "sampler", sampler)
    _set_input(node, "scheduler", scheduler)


def set_seed(workflow: dict[str, Any], seed: int | None = None) -> int:
    """Set seed on the SeedInput node. If seed is None, randomize. Returns the seed used."""
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    found = _find_node_by_class(workflow, "SeedInput")
    if found is None:
        raise ValueError("SeedInput node not found in template")
    _, node = found
    _set_input(node, "seed", seed)
    return seed


def set_load_image(workflow: dict[str, Any], node_id: str, filename: str) -> None:
    """Set the image field on a specific LoadImage node by node id."""
    node = workflow.get(node_id)
    if node is None or node.get("class_type") != "LoadImage":
        raise ValueError(f"LoadImage node '{node_id}' not found")
    node.setdefault("inputs", {})["image"] = filename
