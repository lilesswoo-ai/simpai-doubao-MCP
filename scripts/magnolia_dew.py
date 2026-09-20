"""Magnolia dew collecting scene."""
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient

LORA_HANFU = "Krea2\\新中式汉服neo_tangzhuang-KreaRaw.safetensors"

PROMPT = (
    "Chinese giant aesthetics, 16:9 cinematic, photorealistic, hyperreal. "
    "After-rain morning in a giant white magnolia forest. "
    "Two adult Hanfu women on a wide curved giant magnolia petal platform collecting dew. "
    "Camera inside the petal, 35mm lens, eye-level medium shot, both figures fully visible. "
    "Standing woman at right, occupies 55% frame height, wearing apricot crossed-collar top and dark teal long skirt, "
    "soft silk with fine weave and natural drape, hair in simple bun, holding a small bamboo basket, "
    "other hand gently touching the giant petal edge, leaning forward, looking at companion. "
    "Kneeling woman behind center, occupies 35% frame height, wearing grey-lotus-purple Hanfu and cream sheer shawl, "
    "crouching in the petal hollow, holding a small celadon cup catching dew drops from petal tip. "
    "Natural East Asian adult faces, real skin texture, normal proportions, feet and skirt resting on petal with contact shadows. "
    "Only white magnolia as the plant: thick woody branches, thick oval leaves, large cup-shaped creamy white flowers with faint pink base. "
    "One giant magnolia flower over 10 meters across supports them, only local petal surface visible, curved petal forms platform. "
    "Left side: a two-story-tall creamy white petal rises cropped at top. Right top: another giant petal like a canopy. "
    "Diagonal woody branches recede into overlapping flowers and leaves. Giant plants extend from feet to above frame. "
    "Foreground petal edge slight occlusion, main activity clear. "
    "Sun at left rear, low warm backlight through petal, creamy translucent with fine pink, longitudinal veins and creases visible. "
    "Cool sky fill on faces, rim light on hair and sheer silk. Dappled branch shadows, deep shadows with detail, no blown highlights. "
    "Small normal-sized dew drops with real refraction, a string of tiny water drops being collected. "
    "Rough bark with cracks and moss, magnolia leaves with clear veins and waxy surface. "
    "Same focal plane on figures and petal, medium depth of field, background slightly soft. "
    "Cream, grey-green, dark teal, apricot, lotus-purple low saturation. Very light after-rain mist, clear and transparent. "
    "Grand, intimate, serene Eastern poetic life. "
    "No lotus, no third person, no plastic, no CG, no illustration, no painting, no over-smoothed skin, no HDR glow, no fake god rays, no text, no watermark."
)
NEG = "lotus, third person, cartoon, anime, illustration, painting, cgi, plastic skin, hdr bloom, fake god rays, glowing, text, watermark, deformed, extra fingers, floating, tiny people, small plants, over-sharpened"

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        print("Generating magnolia dew scene...")
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
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["20", 1], "text": PROMPT}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["20", 1], "text": NEG}},
            "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 688, "batch_size": 1}},
            "9": {"class_type": "KSampler", "inputs": {
                "model": ["20", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["8", 0], "seed": 77,
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
            out = Path(r"I:\SimpAI\users\Local\outputs-mcp") / "magnolia_dew_2women.png"
            out.write_bytes(data)
            print(f"DONE: {out}")
    finally:
        await client.aclose()

asyncio.run(main())
