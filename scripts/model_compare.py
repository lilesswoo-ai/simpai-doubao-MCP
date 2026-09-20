"""Compare 11 models with same prompt/seed, no LoRA, add model name below."""
import asyncio, sys, random
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

MODELS = [
    "Krea2\\krea2_raw_fp8_scaled.safetensors",
    "Krea2\\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors",
    "Krea2\\Krea2-turbo-Ink_Jade-AIO-V1-fp8.safetensors",
    "Krea2\\Krea2-turbo-White_Marble-AIO_V1_FP8.safetensors",
    "Krea2\\moodyKrea2Mix_v40INT8CONVROT.safetensors",
    "Krea2\\黑兽瑟瑟darkBeastINT8Convrot2_krea211INT8Convrot.safetensors",
    "Krea2\\红潮编辑加速redcraft23INT8INT4FP8_2Krea2Edition.safetensors",
    "Krea2\\瑟瑟专用krea2GPTGrandPUSSYTruth_krea2GPT.safetensors",
    "Krea2\\摄影优化rayArtshoot_krea2NSFWV2.safetensors",
    "Krea2\\新版Krea2museByStableYogi_v30TurboInt8.safetensors",
    "Krea2\\亚洲美女1125Krea2AsianUtopian_v10.safetensors",
]

PROMPT = (
    "Chinese giant aesthetics, 16:9 cinematic, photorealistic. "
    "A slender curvy East Asian woman in white hanfu, sheltering under a giant pink cherry blossom branch. "
    "The branch and flowers form a canopy above, raindrops falling through petals. "
    "She sits on a giant petal, knees bent, sheer robes damp, collarbones and legs visible. "
    "Face clear, peaceful expression. Wet hair. "
    "Rain and falling petals, misty background. Soft rainy light, white and pink tones. 8K."
)
NEG = "low quality, blurry, deformed, cartoon, anime, illustration, modern, text, watermark, multiple people, normal scale, huge breasts, flat chest, cgi, plastic, over-smoothed, hdr glow, fake rays, extra fingers, floating, sunny"

SEED = 88888
OUT_DIR = Path(r"I:\SimpAI\users\Local\outputs-mcp\model_compare")
OUT_DIR.mkdir(exist_ok=True)

def get_model_name(path):
    return Path(path).stem

async def gen_one(client, unet, prompt, seed):
    wf = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEG}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 688, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["3", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": seed,
            "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "compare"}},
    }
    pid = await client.submit_prompt(wf)
    for _ in range(180):
        await asyncio.sleep(1)
        if await client.is_done(pid):
            break
    imgs = await client.list_output_images(pid)
    if imgs:
        return await client.view_image(imgs[0]["filename"], imgs[0].get("subfolder", ""), imgs[0].get("type", "output"))
    return None

def add_label(img_bytes, label):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    label_h = 60
    canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 28)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 28)
        except:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    x = (w - tw) // 2
    draw.text((x, h + 15), label, fill=(0, 0, 0), font=font)
    out = io.BytesIO()
    canvas.save(out, format="PNG")
    return out.getvalue()

import io

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        for unet in MODELS:
            name = get_model_name(unet)
            print(f"Generating {name}...")
            sys.stdout.flush()
            data = await gen_one(client, unet, PROMPT, SEED)
            if data:
                labeled = add_label(data, name)
                out = OUT_DIR / f"{name}.png"
                out.write_bytes(labeled)
                print(f"DONE: {out}")
                sys.stdout.flush()
    finally:
        await client.aclose()

asyncio.run(main())
