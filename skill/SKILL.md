---
name: simpai-image-gen
description: 调用本地 SimpAI（ComfyUI）生成/编辑图片。支持 Krea2 文生图/图像编辑/批量分镜、Z-imageT Turbo、MiniMax-H3 R2I 参考图生图、ReActor 换脸、Qwen Image 2.1（说"qwen"时调用）。当用户要求生图、改图、批量出图、分镜图、换脸时使用。
---

# SimpAI 本地生图 Skill

## 环境信息

- SimpAI 根目录：`I:\SimpAI\`
- ComfyUI API：`http://127.0.0.1:8188`（只要 ComfyUI 后端在跑即可，不需要打开 Gradio 前端）
- MCP Server 目录：`I:\SimpAI\mcp-server\`
- Python venv：`I:\SimpAI\mcp-server\.venv\Scripts\python.exe`
- 输出目录：`I:\SimpAI\users\Local\outputs-mcp\`
- 文件名格式：`YYYY-MM-DD_HH-MM-SS_xxxx.png`

## 调用方式

用 venv Python 运行脚本，工作目录设为 `I:\SimpAI\mcp-server`。

### 基本模板

```python
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient
from simpai_mcp.tools import krea2

PROMPT = "你的提示词"

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        r = await krea2.krea2_txt2img(client, prompt=PROMPT, steps=8, width=832, height=1216)
        res = await krea2.krea2_get_result(client, r['prompt_id'], wait=True, timeout=180)
        print('done:', res['images'])
    finally:
        await client.aclose()

asyncio.run(main())
```

---

## 提示词写作规范（吸收 claude-image / prompt-optimizer / unprompted 经验）

### 七条核心原则

1. **意图开头** — 第一句直接说要什么，不绕弯子
2. **六段式结构** — 按下面六段组织，不要堆砌
3. **一个风格锚点** — 只定一个主风格，不要"胶片+赛博+水彩"叠三个
4. **规格语言，不用夸赞语言** — 写"3:4竖构图，832×1216，f/2.8浅景深"，不写"绝美、惊艳、神作"
5. **修图用 change-only / preserve** — 编辑图时明确"只改X，保留Y"
6. **迭代不堆砌** — 用户说"磨皮重一点"，只改磨皮参数，不要顺手加一堆新元素
7. **负面提示词克制** — 10-20 个词足够，不要堆 50 个

### 六段式提示词结构

```
[意图] 一句话说明要生成什么
[场景] 环境、地点、时间、天气
[主体] 人物/物体、位置、姿态
[细节] 服装、表情、道具、材质
[风格] 一个主风格锚点 + 摄影参数
[约束] 画幅、比例、必须保留/必须排除的
```

### 文生图示例（人像）

```
Photorealistic portrait, young East Asian woman sitting by a lakeside pavilion window.
Scene: Jiangnan garden, pond outside the window, soft overcast daylight.
Subject: centered figure, side-sitting, one leg extended, hand resting on windowsill.
Details: black lace silk qipao, pearl earrings, hair in simple bun, delicate makeup.
Style: 85mm medium telephoto, f/2.8 shallow depth, photorealistic, muted elegant tones.
Constraints: 3:4 vertical, focus on face, background softly blurred.
```

### 图像编辑示例（change-only 原则）

```
Change ONLY the background to a watercolor style with simplified composition and white space.
Preserve: the cat's pose, expression, position, and all facial details.
Use soft diffused lighting, watercolor paper texture, minimal background elements.
```

### 批量生成 Wildcards 语法

批量变体用变量占位符，Python 里组合：

```
[scene] = [园林月洞门 | 古街青石板 | 竹林小径 | 荷塘石桌 | 水乡乌篷船]
[pose]  = [面对面拥抱 | 背后环腰 | 坐腿上对视 | 亲吻额头 | 并肩靠肩]
[light] = [黄金时刻暖光 | 清晨晨雾 | 傍晚蓝调 | 月夜柔光 | 雨后漫射]
```

组合 N 张时用 `itertools.product` 或随机抽样，每张独立 seed。

---

## 可用模型与参数

### 1. Krea2（默认，最快）

- 模型：`Krea2\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors`
- CLIP：`qwen3vl_4b_fp8_scaled.safetensors`
- VAE：`qwen_image_vae.safetensors`
- 采样：euler / simple, steps=8, cfg=1.0
- 竖版：832×1216，横版：1216×832，9:16 竖版：832×1472

**工具函数：**
- `krea2.krea2_txt2img(client, prompt, width, height, steps=8, cfg=1.0, seed=None, loras=None)`
- `krea2.krea2_edit_image(client, prompt, image_path, denoise=0.6, ...)`
- `krea2.krea2_get_result(client, prompt_id, wait=True, timeout=180)`

**denoise 经验：**
- 风格迁移：0.35–0.5（保留原图内容）
- 改内容/换物体：0.6–0.8
- 完全重画：0.9–1.0

### 2. Z-imageT Turbo

- 模型：`z_image_turbo_int8_convrot.safetensors`
- CLIP：`qwen_3_4b.safetensors`（type=lumina2）
- VAE：`ae.safetensors`
- 采样：euler_ancestral / beta, steps=8, cfg=1.0

### 3. MiniMax-H3 R2I（高质量，慢）

- 模型：20GB+15GB，首次加载需几分钟
- UNET：`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`
- CLIP：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（type=minimax）
- VAE：`minimax_h3_video_vae_int8_convrot.safetensors`
- LoRA：`minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`
- 采样：er_sde / beta57, steps=10, cfg=1.0
- 尺寸：1024×1024

### 4. ReActor 换脸

- swap_model: `inswapper_128.onnx`
- facedetection: `retinaface_resnet50`
- face_restore_model: `codeformer-v0.1.0.pth`

### 5. Qwen Image 2.1（用户说"qwen"时调用）

**触发词**：用户消息中出现"qwen"或"用qwen"时，使用本模型而非默认 Krea2。
**识别参考图**：如果用户上传了图片，自动作为图像编辑的参考图；没上传则为文生图。

**文生图配置**：
- UNET：`qwen_image_2512_fp8_e4m3fn.safetensors`
- CLIP：`qwen_2.5_vl_7b_fp8_scaled.safetensors`（CLIPLoader type=qwen_image）
- VAE：`qwen_image_vae.safetensors`
- 采样：euler / beta, steps=8, cfg=1.0
- 横版16:9：1280×720，竖版3:4：864×1216

**图像编辑配置**：
- UNET：`qwen_image_edit_2511_fp8mixed.safetensors`
- CLIP/VAE 同上
- 自定义节点：`TextEncodeQwenImageEditPlusPro_lrzjason`（输入 clip/vae/image1~5/prompt，输出 conditioning + latent）
- KSampler 的 positive 和 negative 都接该节点的 conditioning 输出，latent 接该节点的 latent 输出

**调用脚本**：`I:\SimpAI\mcp-server\qwen21_workflow.py` 中的 `make_t2i_workflow()` 和 `make_edit_workflow()`。

---

## 常用负面提示词

```
low quality, blurry, deformed, extra limbs, extra fingers,
cartoon, anime, illustration, 3d render,
watermark, text, signature,
ugly, overexposed, underexposed, bad anatomy
```

## 批量生成

用 `krea2.krea2_batch_generate(client, shots=[...], ...)` 提交批量任务，每张约 25 秒。

## 参考图规则

- **用户上传的图片自动作为编辑参考图**：当用户在对话中上传了图片，且要求修改/编辑/换风格时，必须用上传的图片作为参考图（image input），而不是重新文生图。
- 参考图处理流程：
  1. 把用户上传的图片复制到 ComfyUI input 目录：`I:\SimpAI\SimpAIStudiowin\SimpAI_Studio\comfy\input\`
  2. 用图像编辑工作流（Krea2 edit / Qwen edit）加载该图作为输入
  3. 提示词中用 "Preserve: ..." 明确保留不需要改动的部分
- 如果用户没有上传图片但要求编辑，需要询问用户参考图在哪里。

## 交付要求

- 生成后用 `present_files` 把图片展示给用户
- 图片路径：`I:\SimpAI\users\Local\outputs-mcp\*.png`
- 临时脚本用完删除
