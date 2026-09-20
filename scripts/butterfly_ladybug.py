"""4 scenes: brighter background, ladybug only, woman riding or kicking ladybug."""
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient

LORA_HANFU = "Krea2\\新中式汉服neo_tangzhuang-KreaRaw.safetensors"
LORA_ATMO = "Krea2\\氛围摄影atmospheric photography.safetensors"
NEG = "low quality, blurry face, deformed, cartoon, anime, illustration, modern, text, watermark, multiple people, normal scale, bird wings, feathered wings, angel wings, huge breasts, standing, dark background, aphids, green bugs"

SCENES = [
    ("ride_ladybug",
     "Chinese giant aesthetics, 16:9, photorealistic macro photography, bright soft natural light. "
     "A slender woman in pale pink hanfu skirt sitting on the back of a giant seven-spot ladybug, "
     "red shell with black spots, the size of a small horse. The ladybug crawls on a giant rose petal. "
     "On her back grow FOUR DELICATE BUTTERFLY WINGS covered in iridescent scales, open gently. "
     "A small silk shoulder bag slung over her shoulder. One hand grips the ladybug's antenna, "
     "the other hand holds a small pollen pouch. Bare legs and feet visible, one foot dangling. "
     "Face clearly visible, smiling. Bright garden light, pink and green tones. 8K."),
    ("kick_ladybug",
     "Chinese giant aesthetics, 16:9, photorealistic macro photography, bright soft natural light. "
     "A slender woman in pale blue hanfu skirt, flying just above a giant seven-spot ladybug, "
     "red shell with black spots, the size of a small dog. The ladybug crawls on a giant clover leaf. "
     "On her back grow FOUR BLUE BUTTERFLY WINGS vibrating, motion blur. "
     "A small silk shoulder bag slung over her shoulder, her hand grips the strap. "
     "One bare foot gently kicks the ladybug's shell playfully. "
     "Skirt flowing. Face clearly visible, laughing. Bright daylight, fresh green and blue tones. 8K."),
    ("ladybug_flower",
     "Chinese giant aesthetics, 16:9, photorealistic macro photography, bright soft natural light. "
     "A slender woman in pale yellow hanfu skirt sitting on a giant seven-spot ladybug, "
     "red shell with black spots, the size of a pony. The ladybug sits on a giant sunflower petal. "
     "On her back grow FOUR YELLOW BUTTERFLY WINGS, open gently. "
     "A small silk shoulder bag slung over her shoulder. One hand touches the ladybug's head, "
     "the other hand collects pollen. Bare legs crossed. "
     "Face clearly visible. Bright golden light, yellow and green tones. 8K."),
    ("ladybug_leaf",
     "Chinese giant aesthetics, 16:9, photorealistic macro photography, bright soft natural light. "
     "A slender woman in white hanfu skirt flying beside a giant seven-spot ladybug, "
     "red shell with black spots, the size of a large dog. The ladybug crawls on a giant leaf. "
     "On her back grow FOUR GOLDEN BUTTERFLY WINGS vibrating, motion blur. "
     "A small silk shoulder bag slung over her shoulder, her hand holds the strap. "
     "One bare foot playfully touches the ladybug's shell. "
     "Skirt flowing. Face clearly visible. Soft morning light, green and gold tones. 8K."),
]

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        for name, prompt in SCENES:
            print(f"Generating {name}...")
            sys.stdout.flush()
            wf = {
                "3": {"class_type": "UNETLoader", "inputs": {
                    "unet_name": "Krea2\\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors",
                    "weight_dtype": "default"
                }},
                "4": {"class_type": "CLIPLoader", "inputs": {
                    "clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2"
                }},
                "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
                "20": {"class_type": "LoraLoader", "inputs": {
                    "model": ["3", 0], "clip": ["4", 0],
                    "lora_name": LORA_HANFU, "strength_model": 1.0, "strength_clip": 1.0
                }},
                "21": {"class_type": "LoraLoader", "inputs": {
                    "model": ["20", 0], "clip": ["20", 1],
                    "lora_name": LORA_ATMO, "strength_model": 1.0, "strength_clip": 1.0
                }},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["21", 1], "text": prompt}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["21", 1], "text": NEG}},
                "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 688, "batch_size": 1}},
                "9": {"class_type": "KSampler", "inputs": {
                    "model": ["21", 0], "positive": ["6", 0], "negative": ["7", 0],
                    "latent_image": ["8", 0], "seed": hash(name) % 100000,
                    "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
                }},
                "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
                "16": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "simpai_mcp"}},
            }
            pid = await client.submit_prompt(wf)
            for _ in range(180):
                await asyncio.sleep(1)
                if await client.is_done(pid):
                    break
            imgs = await client.list_output_images(pid)
            if imgs:
                img = imgs[0]
                data = await client.view_image(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
                from pathlib import Path
                out = Path(r"I:\SimpAI\users\Local\outputs-mcp") / f"lady_{name}.png"
                out.write_bytes(data)
                print(f"DONE {name}: {out}")
                sys.stdout.flush()
    finally:
        await client.aclose()

asyncio.run(main())
