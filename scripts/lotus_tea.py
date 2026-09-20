"""Detailed giant lotus scene per user prompt."""
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient

LORA_HANFU = "Krea2\\新中式汉服neo_tangzhuang-KreaRaw.safetensors"
LORA_ATMO = "Krea2\\氛围摄影atmospheric photography.safetensors"

PROMPT = (
    "Chinese giant aesthetics, 16:9 cinematic, photorealistic, hyperreal macro. "
    "Cinematic shot on a giant lotus leaf platform, 35mm lens, eye-level medium shot, "
    "two adult Hanfu women sitting on the leaf drinking tea, each figure occupies about 40% of frame height. "
    "Main figure at right third, in moon-white silk Hanfu; second figure behind center, in pale cyan Hanfu. "
    "Natural East Asian adult faces, real skin texture, soft skin tones, dark hair in simple buns, "
    "fingers naturally holding tea cups, gazing at tea bowls. "
    "Sheer silk sleeves with translucent fibers, silk folds draping by gravity, skirt spreading on curved leaf surface. "
    "Small tea tray and two ceramic cups sized proportionally, real contact shadows. "
    "Giant lotus leaf 20 meters across, only part occupies bottom of frame, wide green surface with veins extending far. "
    "On the left, a giant pink-white lotus flower three stories tall, only 2-3 huge petals visible, "
    "curved petals rising from water, top and left edges cropped out of frame. "
    "Giant lotus leaf above like a canopy, underside veins visible, leaf edge cropped at top. "
    "Thick lotus stems behind. Giant plants cover 75% of frame but faces clear. "
    "Early morning after rain, warm low backlight from left rear through lotus leaves, "
    "petals translucent showing longitudinal fibers, dappled light on leaf and shoulders, "
    "rim light on hair and sheer silk, sky fill on faces, water reflection below. "
    "Real waxy lotus leaf surface, dense veins, natural curls, small dew drops with real refraction. "
    "Petals with fine longitudinal veins, creamy to pale pink gradient, translucent edges. "
    "Medium depth of field, background soft but visible. Green and lotus-pink tones, low saturation. "
    "Small water surface with ripples and reflections at bottom. "
    "Cinematic still, fine natural texture, correct perspective, grand yet intimate, elegant serene. "
    "No CG plastic look, no game render, no illustration, no painting, no over-smoothed skin, no HDR glow."
)
NEG = "cartoon, anime, illustration, painting, cgi, plastic skin, hdr bloom, fake god rays, glowing, text, watermark, deformed, extra fingers, floating, tiny people, small plants, full flower in frame, multiple people far away"

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        print("Generating lotus tea scene...")
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
                "latent_image": ["8", 0], "seed": 42,
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
            out = Path(r"I:\SimpAI\users\Local\outputs-mcp") / "lotus_tea_2women.png"
            out.write_bytes(data)
            print(f"DONE: {out}")
    finally:
        await client.aclose()

asyncio.run(main())
