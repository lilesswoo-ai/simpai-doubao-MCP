"""Wrapper to run SimpAI MCP over streamable-http transport."""
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simpai_mcp.server import mcp, BASE_URL

print(f"[SimpAI MCP] Connecting to ComfyUI at {BASE_URL}")
print(f"[SimpAI MCP] HTTP server on http://127.0.0.1:8765/mcp")

mcp.settings.host = "127.0.0.1"
mcp.settings.port = 8765
mcp.run(transport="streamable-http")
