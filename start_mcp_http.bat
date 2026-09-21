@echo off
title SimpAI MCP HTTP Server
cd /d "%~dp0"
set COMFY_LOCAL_URL=http://127.0.0.1:8188
set SIMPAI_COMFYD_URL=http://127.0.0.1:8188
echo Starting SimpAI MCP HTTP on http://127.0.0.1:8765/mcp
echo Make sure SimpAI ComfyUI (port 8188) is running first.
echo.
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe comfy_mcp_http.py
) else (
    python comfy_mcp_http.py
)
pause
