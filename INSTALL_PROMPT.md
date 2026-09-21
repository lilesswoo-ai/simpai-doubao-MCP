# SimpAI MCP Server 自动安装配置提示词

把下面整段复制给 agent 执行（先把 `你的SimpAI路径` 替换成实际路径）：

---

请帮我完成 SimpAI MCP Server 的安装、配置和测试，按以下步骤执行：

SimpAI 安装路径：你的SimpAI路径
（例如 I:\SimpAI，下面所有操作都基于这个路径）

## 1. 确认目录结构

MCP 代码应该在 `{SimpAI路径}\mcp-server\` 下。检查以下文件是否齐全：
- `simpai_mcp\server.py`
- `simpai_mcp\client.py`
- `simpai_mcp\tools\krea2.py`
- `comfy_mcp_http.py`
- `start_mcp_http.bat`
- `requirements.txt`

## 2. 创建 Python 虚拟环境

在 `{SimpAI路径}\mcp-server\` 下执行：

```powershell
py -3.11 -m venv .venv
```

如果 `.venv` 已存在则跳过。

## 3. 安装依赖

```powershell
cd {SimpAI路径}\mcp-server
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## 4. 配置环境变量

检查 `.env` 文件是否存在。如果不存在，从 `.env.example` 复制：

```powershell
copy .env.example .env
```

## 5. 验证 ComfyUI 连接

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8188/api/system_stats" -UseBasicParsing -TimeoutSec 5
```

如果失败，询问用户是否需要启动 ComfyUI：
```powershell
Start-Process -FilePath "{SimpAI路径}\SimpAIStudiowin\run_ComfyUI_工作流模式.bat" -WorkingDirectory "{SimpAI路径}\SimpAIStudiowin" -WindowStyle Minimized
```
轮询等待 8188 端口就绪（每 5 秒，最多 5 分钟）。

## 6. 端到端测试

```powershell
cd {SimpAI路径}\mcp-server
.\.venv\Scripts\python.exe -c "import asyncio,sys; sys.path.insert(0,'.'); from simpai_mcp.client import ComfyClient; from simpai_mcp.tools import krea2; async def m():
 c=ComfyClient('http://127.0.0.1:8188')
 try:
  r=await krea2.krea2_txt2img(c,prompt='a red flower',steps=8,width=832,height=1216)
  print('submitted:',r['prompt_id'])
  res=await krea2.krea2_get_result(c,r['prompt_id'],wait=True,timeout=180)
  print('status:',res['status'])
 finally: await c.aclose()
asyncio.run(m())"
```

## 7. 启动 MCP HTTP 服务

```powershell
cd {SimpAI路径}\mcp-server
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "comfy_mcp_http.py" -WindowStyle Minimized
```

服务运行在 `http://127.0.0.1:8765/mcp`

## 8. 汇报结果

- 哪些步骤成功了
- 测试图的完整路径
- MCP 服务是否在运行
- 如果有任何失败，说明原因和解决方案
