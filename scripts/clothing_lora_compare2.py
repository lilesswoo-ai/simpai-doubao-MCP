"""Compare 7 clothing LoRAs with cherry blossom prompt, same seed/prompt."""
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(r"I:\SimpAI\users\Local\outputs-mcp\clothing_compare2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "Krea2\\摄影优化rayArtshoot_krea2NSFWV2.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
SEED = 88888
PROMPT_TEMPLATE = ("Chinese giant aesthetics, 16:9, photorealistic cinematic photography, rainy cherry blossom garden. "
          "A beautiful young East Asian woman with long wet black hair in a simple updo, sitting on a giant pink cherry blossom petal platform the size of a bed. "
          "She wears {clothing}, wet from rain, clinging to her curvy body, deep cleavage visible. "
          "Her left hand reaches up and gently holds a cherry blossom branch above her, water droplets falling from the petals. "
          "Rain falls softly, pink cherry blossoms drift through the air, giant cherry petals surround her like a sea of pink. "
          "Misty rain background, soft diffused overcast light, pink and white tones, wet skin glistening. "
          "Full body visible, bare legs and feet on the petal platform. 8K, ultra detailed, photorealistic.")
NEG = "low quality, blurry, deformed, cartoon, anime, illustration, modern clothing, text, watermark, multiple people, normal scale, bird wings, dark, ugly, bad hands, extra fingers"

LORAS = [
    ("中国古代服饰yijinggufeng", "Krea2\\中国古代服饰yijinggufeng.safetensors", "无",
     "a sheer white semi-transparent ancient Chinese robe"),
    ("中国古代服饰肚兜guzhuang", "Krea2\\中国古代服饰肚兜guzhuang.safetensors", "无",
     "a sheer red Chinese bellyband (dudou) with bare shoulders and back"),
    ("包臀裙btq", "Krea2\\包臀裙btq-nickBodyconskirtE58C85E8.mbqw.safetensors", "无",
     "a tight fitted bodycon mini skirt, sheer fabric"),
    ("新中式旗袍neo_cheongsam", "Krea2\\新中式旗袍neo_cheongsam-KreaTurob.safetensors", "无",
     "a sheer white semi-transparent modern Chinese cheongsam (qipao)"),
    ("新中式汉服neo_tangzhuang", "Krea2\\新中式汉服neo_tangzhuang-KreaRaw.safetensors", "neo_tangzhuang",
     "a sheer white semi-transparent neo Chinese hanfu robe"),
    ("旗袍cheongsam2", "Krea2\\旗袍cheongsam Nick_cheongsam2_Krea2.safetensors", "无",
     "a form-fitting Chinese cheongsam (qipao) with high side slit"),
    ("高贵汉服TFX", "Krea2\\高贵汉服!krea2_TFXChineseStyleHanfu_V1.0.safetensors", "无",
     "an elegant noble Chinese hanfu with flowing silk layers"),
]

async def gen_one(client, lora_path, lora_name, trigger, clothing):
    prompt_text = PROMPT_TEMPLATE.format(clothing=clothing)
    if trigger and trigger != "无":
        prompt_text = trigger + ", " + prompt_text
    wf = {
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "krea2"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "20": {"class_type": "LoraLoader", "inputs": {
            "model": ["3", 0], "clip": ["4", 0],
            "lora_name": lora_path, "strength_model": 1.0, "strength_clip": 1.0
        }},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["20", 1], "text": prompt_text}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["20", 1], "text": NEG}},
        "8": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 688, "batch_size": 1}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["20", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["8", 0], "seed": SEED,
            "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0
        }},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["5", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "clothing_cmp2"}},
    }
    pid = await client.submit_prompt(wf)
    for _ in range(180):
        await asyncio.sleep(1)
        if await client.is_done(pid):
            break
    imgs = await client.list_output_images(pid)
    if not imgs:
        return None
    img = imgs[0]
    data = await client.view_image(img["filename"], img.get("subfolder", ""), img.get("type", "output"))
    p = OUT_DIR / f"{lora_name}.png"
    p.write_bytes(data)
    return p

def make_collage(results):
    font_title = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 48)
    font_label = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)
    font_small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)

    tile_w = 1216
    tile_h = 688 + 70
    cols = 2
    rows = (len(results) + cols - 1) // cols
    header_h = 220
    margin = 40

    total_w = margin * 2 + tile_w * cols + margin * (cols - 1)
    total_h = header_h + margin + tile_h * rows + margin * (rows - 1) + margin

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, 30), "Krea2 服装 LoRA 效果对比（樱花雨）", fill="black", font=font_title)
    info = (f"模型: rayArtshoot_krea2NSFWV2  |  Seed: {SEED}  |  采样: euler/simple 8步 cfg=1.0  |  "
            f"尺寸: 1216x688  |  LoRA强度: 1.0")
    draw.text((margin, 100), info, fill="#333", font=font_small)
    draw.text((margin, 130), f"提示词模板: {PROMPT_TEMPLATE[:130]}...", fill="#666", font=font_small)
    draw.text((margin, 160), f"负面: {NEG[:100]}...", fill="#666", font=font_small)

    for i, (label, path, trigger) in enumerate(results):
        r = i // cols
        c = i % cols
        x = margin + c * (tile_w + margin)
        y = header_h + margin + r * (tile_h + margin)
        if path and path.exists():
            img = Image.open(path).convert("RGB").resize((tile_w, 688))
            canvas.paste(img, (x, y))
        else:
            draw.rectangle([x, y, x+tile_w, y+688], fill="#eee")
        draw.text((x+10, y+695), f"模型: rayArtshoot_krea2NSFWV2", fill="#333", font=font_small)
        draw.text((x+10, y+720), f"LoRA: {label}", fill="#c00", font=font_small)

    out = Path(r"I:\SimpAI\users\Local\outputs-mcp\clothing_lora_compare2.png")
    canvas.save(out, quality=92)
    print(f"Collage saved: {out}")
    return out

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    results = []
    try:
        for name, path, trigger, clothing in LORAS:
            print(f"Generating {name}...")
            sys.stdout.flush()
            p = await gen_one(client, path, name, trigger, clothing)
            results.append((name, p, trigger))
            print(f"  -> {p}")
            sys.stdout.flush()
    finally:
        await client.aclose()
    collage = make_collage(results)
    print(f"Done!")

asyncio.run(main())
