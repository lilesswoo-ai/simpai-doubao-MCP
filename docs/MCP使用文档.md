# SimpAI MCP 使用文档

本文件包含 MCP 的安装接入、工具说明、调用方式和生图 Skill 完整说明。

---

## 一、MCP 简介

SimpAI MCP 是独立于 SimpAI 的控制模块，通过 ComfyUI HTTP API（默认 `http://127.0.0.1:8188`）操控本地模型生图。

**设计原则：**
- 完全独立文件夹，不改动 SimpAI 任何文件（自动更新不受影响）
- 只调 HTTP API，不 import SimpAI 源码
- 有独立 Python venv，依赖隔离
- 输出统一存到 `I:\SimpAI\users\Local\outputs-mcp\`

---

## 二、安装接入（新电脑）

### 前提条件

1. 已安装 SimpAI 主程序
2. 已下载所需模型（Krea2 / Z-imageT / MiniMax-H3 / ReActor 等）
3. SimpAI 已启动，ComfyUI API 正常（浏览器打开 `http://127.0.0.1:8188/api/system_stats` 能看到 JSON）
4. 安装了 Python 3.11

### 步骤

**1. 复制 mcp-server 文件夹**

把整个 `mcp-server` 文件夹复制到目标位置（不要复制 `.venv` 和 `output`）。

**2. 创建虚拟环境**

```powershell
cd C:\path\to\mcp-server
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install mcp httpx pydantic python-dotenv
```

**3. 配置环境变量**

```powershell
copy .env.example .env
```

编辑 `.env`：
```
SIMPAI_COMFYD_URL=http://127.0.0.1:8188
```

**4. 配置 MCP 客户端**

在 MCP 客户端（豆包 / Claude Desktop / Cherry Studio）配置文件里加入：

```json
{
  "mcpServers": {
    "simpai": {
      "command": "C:\\path\\to\\mcp-server\\.venv\\Scripts\\python.exe",
      "args": ["-m", "simpai_mcp.server"],
      "cwd": "C:\\path\\to\\mcp-server",
      "env": {
        "SIMPAI_COMFYD_URL": "http://127.0.0.1:8188"
      }
    }
  }
}
```

> 路径用双反斜杠 `\\`。

**5. 验证**

调用 `krea2_txt2img` 生成一张测试图，成功即完成。

---

## 三、工具清单

| 工具 | 作用 |
|------|------|
| `krea2_txt2img` | Krea2 文生图，支持 LoRA |
| `krea2_edit_image` | Krea2 图像编辑（上传本地图 + 指令 + denoise） |
| `krea2_get_result` | 轮询任务并下载结果图 |
| `krea2_batch_generate` | 批量提交多个分镜 |
| `krea2_batch_status` | 查询批量进度 |
| `krea2_batch_results` | 获取批量全部结果 |
| `zimage_txt2img` | Z-imageT Turbo 文生图 |
| `zimage_get_result` | Z-imageT 结果下载 |
| `reactor_swap_face` | ReActor 换脸 |
| `reactor_get_result` | 换脸结果下载 |
| `h3_r2i_generate` | MiniMax-H3 R2I 参考图生图 |
| `h3_get_result` | H3 结果下载 |

---

## 四、已验证模型参数

### Krea2（默认，最快）

- UNET: `Krea2\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors`
- CLIP: `qwen3vl_4b_fp8_scaled.safetensors`（type="krea2"）
- VAE: `qwen_image_vae.safetensors`
- 采样: euler / simple, steps=8, cfg=1.0
- 竖版: 832×1216，横版: 1216×832

> 注意：模型名必须用反斜杠 `\`，不能用正斜杠。

### Z-imageT Turbo

- UNET: `z_image_turbo_int8_convrot.safetensors`
- CLIP: `qwen_3_4b.safetensors`（type="lumina2"）
- VAE: `ae.safetensors`
- 采样: euler_ancestral / beta, steps=8, cfg=1.0

### MiniMax-H3 R2I（高质量，慢）

- UNET: `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`（20GB）
- CLIP: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（type=minimax）
- VAE: `minimax_h3_video_vae_int8_convrot.safetensors`
- LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`
- 采样: er_sde / beta57, steps=10, cfg=1.0
- 尺寸: 1024×1024
- 首次加载需几分钟

### ReActor 换脸

- swap_model: `inswapper_128.onnx`
- facedetection: `retinaface_resnet50`
- face_restore_model: `codeformer-v0.1.0.pth`

---

## 五、图像编辑 denoise 经验

| denoise 值 | 效果 | 用途 |
|---|---|---|
| 0.35–0.5 | 保留原图内容，只改风格/质感 | 风格迁移、水彩化 |
| 0.6–0.7 | 保留大致结构，改局部内容 | 换背景、调色、去杂物 |
| 0.8–1.0 | 几乎重新生成 | 完全换内容 |

---

## 六、输出文件

- 输出目录：`I:\SimpAI\users\Local\outputs-mcp\`
- 文件名格式：`YYYY-MM-DD_HH-MM-SS_xxxx.png`
- 批量任务：`outputs-mcp\batches\batch_xxx\`

---

## 七、生图 Skill（对话调用模板）

把以下内容存为 Skill，在豆包对话中直接说"用 simpai 生图"即可触发：

```markdown
---
name: simpai-image-gen
description: 调用本地 SimpAI（ComfyUI）生成/编辑图片。支持 Krea2 文生图/图像编辑/批量分镜、Z-imageT Turbo、MiniMax-H3 R2I 参考图生图、ReActor 换脸。当用户要求生图、改图、批量出图、分镜图、换脸时使用。
---

# SimpAI 本地生图 Skill

## 环境信息

- SimpAI 根目录：I:\SimpAI\
- ComfyUI API：http://127.0.0.1:8188
- MCP Server 目录：I:\SimpAI\mcp-server\
- Python venv：I:\SimpAI\mcp-server\.venv\Scripts\python.exe
- 输出目录：I:\SimpAI\users\Local\outputs-mcp\
- 文件名格式：YYYY-MM-DD_HH-MM-SS_xxxx.png

## 基本调用模板

用 venv Python 运行脚本，工作目录设为 mcp-server。

### 文生图

```python
import asyncio, sys
sys.path.insert(0, r'I:\SimpAI\mcp-server')
from simpai_mcp.client import ComfyClient
from simpai_mcp.tools import krea2

async def main():
    client = ComfyClient('http://127.0.0.1:8188')
    try:
        r = await krea2.krea2_txt2img(client, prompt="你的提示词", width=832, height=1216)
        res = await krea2.krea2_get_result(client, r['prompt_id'], wait=True, timeout=180)
        print('done:', res['images'])
    finally:
        await client.aclose()

asyncio.run(main())
```

### 图像编辑

```python
r = await krea2.krea2_edit_image(
    client,
    prompt="改成水彩画风格，背景简化留白",
    image_path=r"C:\path\to\input.png",
    denoise=0.6
)
```

## 提示词格式

人像按以下结构写效果更好：

```
主题：[一句话概括]
主体：[人物位置，构图]
人物与表情：[面容、皮肤、表情、发型、配饰]
服装与姿势：[服装细节、姿势]
背景与光线：[场景、光源]
构图与相机：[比例、角度、镜头、景深]
质感与风格：[写实质感、分辨率、色调]
```

## 批量生成

用 `krea2.krea2_batch_generate` 提交多个分镜，每张约 25 秒。
```

---

## 八、常见问题

**Q: 需要打开 Gradio 前端（8186）吗？**
A: 不需要。只要 ComfyUI 后端（8188）在跑就行。

**Q: SimpAI 更新后 MCP 会坏吗？**
A: 不会。MCP 是独立文件夹，只通过 HTTP API 调用，不改动 SimpAI 文件。

**Q: 模型名用正斜杠报错？**
A: Krea2 的 UNET 模型名必须用反斜杠 `Krea2\...`，不能用正斜杠。

**Q: denoise 设多少合适？**
A: 风格迁移 0.35-0.5，改内容 0.6-0.8，完全重画 0.9+。
