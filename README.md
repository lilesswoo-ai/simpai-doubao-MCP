# SimpAI Doubao MCP

通过 MCP (Model Context Protocol) 让豆包 AI 直接操控本地 SimpAI / ComfyUI 进行图像生成、编辑和批量处理。

## 功能

- **文生图**：Krea2 系列模型（11个底座模型可切换），支持自定义 LoRA
- **图生图 / 图像编辑**：Krea2 图像编辑工作流，支持 denoise 调节
- **批量生成**：锁定 seed/提示词/参数，批量对比不同模型或 LoRA 效果
- **角色一致性**：先生成角色参考图，再用图生图保持人物一致
- **LoRA 管理**：自动扫描本地 LoRA，生成 HTML 参考手册（含触发词）
- **独立部署**：MCP 服务完全独立，不修改 SimpAI 任何文件，更新不冲突

## 架构

```
豆包 AI (Agent)
    ↓ streamable-http (port 8765)
MCP Server (comfy_mcp_http.py)
    ↓ HTTP API
ComfyUI (port 8188)
    ↓
GPU 推理 → 生成图片 → 保存到 outputs-mcp/
```

## 快速开始

### 一键安装（推荐）

把下面这段指令直接发给豆包 AI，**先把 `你的SimpAI路径` 替换成你电脑上 SimpAI 的实际安装路径**（例如 `I:\SimpAI`、`D:\AI\SimpAI`），它会自动完成下载、安装、配置和测试：

```
帮我安装 SimpAI Doubao MCP：

SimpAI 安装路径：你的SimpAI路径
（例如 I:\SimpAI，下面所有操作都基于这个路径）

1. 从 GitHub 克隆仓库：https://github.com/lilesswoo-ai/simpai-doubao-MCP
   克隆到 {SimpAI路径}\mcp-server（如果目录已存在则跳过克隆）
2. 在 {SimpAI路径}\mcp-server 下创建 Python 虚拟环境 .venv
3. 激活虚拟环境并安装依赖：pip install fastmcp httpx pillow
4. 确认 ComfyUI 已在 http://127.0.0.1:8188 运行
   （SimpAI 启动后 ComfyUI API 默认在 8188 端口）
5. 启动 MCP 服务：python comfy_mcp_http.py（运行在 http://127.0.0.1:8765/mcp）
6. 测试：调用 simpai 文生图工具生成一张测试图，确认图片输出到 {SimpAI路径}\users\Local\outputs-mcp\
7. 告诉我在豆包设置里怎么添加 MCP 连接器

注意：不要修改 SimpAI 安装目录下除 mcp-server 以外的任何文件，
所有 MCP 相关操作都在 {SimpAI路径}\mcp-server 内完成，避免 SimpAI 更新时冲突。
```

### 手动安装

### 1. 环境要求

- SimpAI 已安装并运行 ComfyUI（默认端口 8188）
- Python 3.10+
- 豆包客户端（支持 MCP 连接器）

### 2. 安装

```bash
# 克隆仓库
git clone https://github.com/lilesswoo-ai/simpai-doubao-MCP.git
cd simpai-doubao-MCP

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装依赖
pip install fastmcp httpx pillow
```

### 3. 启动 MCP 服务

```bash
# Windows
start_mcp_http.bat

# 或手动
python comfy_mcp_http.py
```

服务运行在 `http://127.0.0.1:8765/mcp`

### 4. 豆包配置 MCP

1. 打开豆包 → 设置 → MCP 连接器
2. 添加 MCP Server，URL 填 `http://127.0.0.1:8765/mcp`
3. 选择 streamable-http transport
4. 保存并启用

## 使用示例

### 文生图

```
用 simpai 生成一张图：
主题：水榭窗边静坐的墨色旗袍女子
竖构图 3:4，照片级写实，8K
```

### 批量对比

```
锁定 seed 和提示词，依次用 7 个服装 LoRA 各生成一张图，
拼成 2 列多行的对比大图
```

### 角色一致性

```
先生成聂小倩的角色参考图（左头像右全身），
然后用这张图做参考，生成 20 张分镜图
```

## 目录结构

```
├── simpai_mcp/           # MCP 核心代码
│   ├── client.py         # ComfyUI HTTP 客户端
│   └── server.py         # FastMCP 工具定义
├── scripts/              # 批量生成脚本示例
│   ├── model_compare.py        # 11 个模型对比
│   ├── clothing_lora_compare2.py  # 服装 LoRA 对比
│   └── gen_lora_manual_v2.py     # LoRA 手册生成
├── docs/
│   ├── MCP使用文档.md     # 详细使用说明
│   ├── lora-manual.html   # Krea2 LoRA 参考手册
│   └── design.html        # 影楼修图工作站设计方案
├── comfy_mcp_http.py      # MCP HTTP 包装（streamable-http）
└── start_mcp_http.bat    # Windows 启动脚本
```

## 已验证的 Krea2 模型配置

| 组件 | 文件 |
|------|------|
| UNET | `Krea2\krea2MuseByStable_v15TurboFp8-INT8_CONVROT.safetensors` |
| CLIP | `qwen3vl_4b_fp8_scaled.safetensors` |
| VAE | `qwen_image_vae.safetensors` |
| 采样 | euler / simple, 8 steps, cfg=1.0 |
| 横版 16:9 | 1216×688 |

### 11 个可切换模型

1. krea2_raw_fp8_scaled
2. krea2MuseByStable_v15TurboFp8
3. Krea2-turbo-Ink_Jade-AIO
4. Krea2-turbo-White_Marble-AIO
5. moodyKrea2Mix_v40
6. 黑兽瑟瑟darkBeast
7. 红潮编辑加速redcraft23
8. 瑟瑟专用krea2GPT
9. 摄影优化rayArtshoot_krea2NSFWV2
10. 新版Krea2museByStableYogi_v30
11. 亚洲美女1125Krea2AsianUtopian

## 输出文件

生成的图片保存在 `{SimpAI路径}\users\Local\outputs-mcp\`，命名格式：
`YYYY-MM-DD_HH-MM-SS_xxxx.png`

## 注意事项

- MCP 服务必须在 SimpAI / ComfyUI 运行时才能使用
- **ComfyUI 未启动时**：Agent 检测到 8188 端口无响应，应先询问用户"是否需要帮你启动 SimpAI/ComfyUI？"，用户确认后执行：
  ```powershell
  Start-Process -FilePath "{SimpAI路径}\SimpAIStudiowin\run_ComfyUI_工作流模式.bat" `
    -WorkingDirectory "{SimpAI路径}\SimpAIStudiowin" -WindowStyle Minimized
  ```
  然后轮询等待 8188 端口就绪（每 5 秒检测一次，最多等 5 分钟），就绪后再继续生图任务。
- 不同电脑部署需要修改 `comfy_mcp_http.py` 中的路径配置
- LoRA 路径使用反斜杠（ComfyUI 要求）
- PIL 中文标签必须用 `msyh.ttc`（微软雅黑），Arial 不支持中文
- rayArtshoot 等纯 UNET 模型必须用 UNETLoader + 单独 CLIPLoader，不能用 CheckpointLoaderSimple

## License

MIT
