"""Regenerate LoRA manual: 3:4 thumbs, better triggers from TSV + filename."""
import os, re, json
from pathlib import Path

LORA_DIR = Path(r"I:\SimpAI\SimpleModels\loras\Krea2")
OUT_HTML = Path(r"I:\SimpAI\mcp-server\ui-design\lora-manual.html")
ASSET_DIR = Path(r"I:\SimpAI\mcp-server\ui-design\lora-thumbs")
ASSET_DIR.mkdir(parents=True, exist_ok=True)

# Parse trigger words CSV (SimpAI native)
trigger_csv = Path(r"I:\SimpAI\SimpleModels\loras\data\lora_trigger_words.csv")
triggers_map = {}
if trigger_csv.exists():
    import csv as csvmod
    with open(trigger_csv, encoding="utf-8", newline="") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            lora_name = row.get("lora_name", "").strip()
            tw = row.get("trigger_word", "").strip()
            if lora_name:
                triggers_map[lora_name] = tw

# Also parse GJJ TSV as secondary
tsv_path = Path(r"I:\SimpAI\SimpAIStudiowin\SimpAI_Studio\comfy\custom_nodes\ComfyUI_GJJ_Nodes\presets\gjj_lora_metadata.tsv")
if tsv_path.exists():
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            lid = parts[0].strip()
            trigger = parts[3].strip()
            if trigger and trigger not in ("", " "):
                triggers_map.setdefault(lid.lower(), trigger)

# Collect safetensors files
safetensors = sorted([f for f in LORA_DIR.iterdir() if f.suffix == ".safetensors"])

def find_thumb(lora_path):
    stem = lora_path.stem
    for ext in [".png", ".webp", ".jpeg", ".jpg"]:
        p = lora_path.with_suffix(ext)
        if p.exists():
            return p
    for f in LORA_DIR.iterdir():
        if f.suffix in [".png", ".webp", ".jpeg", ".jpg"] and f.stem.startswith(stem[:20]):
            return f
    return None

def categorize(name):
    n = name.lower()
    if any(k in n for k in ["nsfw", "blowjob", "doggy", "cowgirl", "pussy", "sex", "thicc", "bbw", "boob", "titty", "banana_tits", "sext", "porn", "erotic", "sexy", "bdsm", "fetish", "sensual", "humnsfw", "yunyun", "humnsfw", "blowjob_epoch", "doggy", "cowgirl", "standingcarry", "pussy_helper", "pussy_inpainting", "grandpussytruth", "mysticxxx", "professional_pornographic", "low_resolution_slider", "klein_epoch"]):
        return "NSFW/性感"
    if any(k in n for k in ["汉服", "旗袍", "cheongsam", "tangzhuang", "hanfu", "clothing", "costume", "dress", "skirt", "btq", "bodycon", "肚兜", "guzhuang", "yijinggufeng", "tfchinese", "hanfu"]):
        return "服装"
    if any(k in n for k in ["face", "脸部", "chinese girl", "asian girl", "hina", "ray versatile", "bfs", "bodyswap", "identity", "character", "三视图", "card", "storyflow", "candidate", "petite", "身材", "thickness", "bbw", "girl", "chinesegirl", "versatile face", "identity_edit", "bfs_v", "bodyswap", "candidate320", "storyflow"]):
        return "人物/角色"
    if any(k in n for k in ["ghibli", "插画", "illustration", "画风", "art style", "royals", "watercolor", "水彩", "komiks", "comic", "harustyle", "bradhamel", "krealumi", "kreaphotoart", "artifact", "kreawatt", "krealeimin", "kreaheleson", "kreashizuka", "kradarrow", "kreacombook", "krawator", "rosy-lofi", "千娇百媚", "izayoi", "krea2muse", "royals", "harustyle", "krealumi", "kreaphotoart", "kreawatt", "krealeimin", "kreaheleson", "kreashizuka", "kradarrow", "kreacombook", "krawator", "rosy-lofi", "izayoi"]):
        return "绘画/插画"
    if any(k in n for k in ["摄影", "photography", "photo", "cinematic", "movie", "film", "atmospheric", "film", "vintage", "70s", "80s", "retro", "fashion", "earth", "animal", "dramatic", "dark", "afterlight", "realism", "real", "ultra_real", "detail", "material", "skin", "light", "golden", "warm", "blue", "purple", "grainy", "herraw", "david dubnitskiy", "russian", "natural", "phone", "low resolution", "klein", "pornographic", "natural earth", "broken_tears", "fashion", "dramatic", "retro", "danish", "tjfilm", "film", "cinematic", "atmospheric", "vintage", "retro", "70s", "80s", "imperfecta", "showa", "rodo", "afterlight", "material", "realism", "real", "ultra", "rdbt", "detail", "tirdb", "photo", "cinematic", "movie", "film", "natural", "earth", "animal", "fashion", "dramatic", "dark", "grainy", "purple", "herraw", "warm", "blue", "golden", "light", "skin", "tone", "color", "photo", "shoot", "ray", "artshoot", "nsfw", "v2", "v3", "v4", "v1", "epoch", "e20", "e40", "e14", "e10", "e50", "step", "step", "int8", "fp8", "convrot", "tensor", "qwen", "sdxl", "flux", "wan", "ltx", "comfy", "krea", "model", "loraholic"]):
        return "摄影"
    if any(k in n for k in ["depth", "control", "turbo", "anything2real", "comic2real", "inpaint", "edit", "assist", "helper", "slider", "mobile", "turbo_lora", "turbo"]):
        return "工具/控制"
    return "其他"

def extract_trigger(name, stem):
    # Check SimpAI CSV first (exact match with backslash)
    for key, trigger in triggers_map.items():
        if stem.lower() in key.lower() or key.lower().endswith(stem.lower()):
            return trigger if trigger else "无"
    # No trigger found
    return "无"

items = []
for f in safetensors:
    thumb = find_thumb(f)
    thumb_rel = ""
    if thumb:
        thumb_name = f.stem + thumb.suffix
        thumb_dest = ASSET_DIR / thumb_name
        if not thumb_dest.exists():
            import shutil
            shutil.copy2(thumb, thumb_dest)
        thumb_rel = f"lora-thumbs/{thumb_name}"
    cat = categorize(f.stem)
    trigger = extract_trigger(f.stem, f.stem)
    items.append({
        "name": f.stem,
        "file": f.name,
        "thumb": thumb_rel,
        "category": cat,
        "trigger": trigger,
    })

cat_order = ["摄影", "绘画/插画", "人物/角色", "服装", "NSFW/性感", "工具/控制", "其他"]
items.sort(key=lambda x: (cat_order.index(x["category"]) if x["category"] in cat_order else 99, x["name"]))

cats = {}
for it in items:
    cats.setdefault(it["category"], []).append(it)

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Krea2 LoRA 参考手册</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Microsoft YaHei", -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { text-align: center; color: #fff; margin: 20px 0 10px; font-size: 28px; }
.stats { text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }
.filter-bar { position: sticky; top: 0; background: #16213e; padding: 12px; text-align: center; z-index: 100; border-radius: 8px; margin-bottom: 20px; }
.filter-bar button { background: #0f3460; color: #fff; border: none; padding: 8px 16px; margin: 4px; border-radius: 20px; cursor: pointer; font-size: 13px; transition: all .2s; }
.filter-bar button:hover, .filter-bar button.active { background: #e94560; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 20px; }
.card { background: #16213e; border-radius: 12px; overflow: hidden; transition: transform .2s; }
.card:hover { transform: translateY(-4px); }
.thumb { width: 100%; aspect-ratio: 3/4; object-fit: cover; background: #0a0a14; display: block; }
.info { padding: 14px; }
.name { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 8px; word-break: break-all; line-height: 1.4; }
.trigger { font-size: 11px; color: #e94560; background: rgba(233,69,96,.1); padding: 4px 8px; border-radius: 4px; display: inline-block; margin-bottom: 8px; word-break: break-all; }
.cat { font-size: 11px; color: #0f3460; background: #e94560; padding: 3px 10px; border-radius: 10px; display: inline-block; }
.cat-section { margin: 30px 0; }
.cat-section h2 { color: #e94560; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-bottom: 16px; font-size: 22px; }
.no-thumb { width: 100%; aspect-ratio: 3/4; display: flex; align-items: center; justify-content: center; background: #0a0a14; color: #555; font-size: 48px; }
</style>
</head>
<body>
<h1>Krea2 LoRA 参考手册</h1>
<div class="stats">共 """ + str(len(items)) + """ 个 LoRA · 分 """ + str(len(cats)) + """ 类</div>
<div class="filter-bar">
<button class="active" onclick="filterAll()">全部</button>
"""

for cat in cat_order:
    if cat in cats:
        html += f'<button onclick="filterCat(\'{cat}\')">{cat} ({len(cats[cat])})</button>\n'

html += """</div>
"""

for cat in cat_order:
    if cat not in cats:
        continue
    html += f'<div class="cat-section" data-cat="{cat}">\n'
    html += f'<h2>{cat} <span style="font-size:14px;color:#888">({len(cats[cat])})</span></h2>\n'
    html += '<div class="grid">\n'
    for it in cats[cat]:
        html += '<div class="card">\n'
        if it["thumb"]:
            html += f'<img class="thumb" src="{it["thumb"]}" loading="lazy" alt="{it["name"]}">\n'
        else:
            html += '<div class="no-thumb">无图</div>\n'
        html += '<div class="info">\n'
        html += f'<div class="name">{it["name"]}</div>\n'
        html += f'<div class="trigger">触发词: {it["trigger"]}</div>\n'
        html += f'<span class="cat">{it["category"]}</span>\n'
        html += '</div></div>\n'
    html += '</div></div>\n'

html += """
<script>
function filterCat(cat) {
    document.querySelectorAll('.cat-section').forEach(s => {
        s.style.display = s.dataset.cat === cat ? '' : 'none';
    });
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    window.scrollTo({top: 0, behavior: 'smooth'});
}
function filterAll() {
    document.querySelectorAll('.cat-section').forEach(s => s.style.display = '');
    document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    window.scrollTo({top: 0, behavior: 'smooth'});
}
</script>
</body>
</html>
"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"Generated: {OUT_HTML}")
print(f"Total LoRAs: {len(items)}")
for cat, items_list in cats.items():
    print(f"  {cat}: {len(items_list)}")
