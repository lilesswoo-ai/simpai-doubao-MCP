@echo off
title SimpAI Comfy-MCP HTTP Server
cd /d "I:\SimpAI\mcp-server"
set COMFY_LOCAL_URL=http://127.0.0.1:8188
echo Starting Comfy MCP HTTP on http://127.0.0.1:8765
echo Make sure SimpAI ComfyUI (port 8188) is running first.
echo.
.\.venv\Scripts\python.exe comfy_mcp_http.py
pause
