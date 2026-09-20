"""Wrapper to run comfy-mcp over streamable-http transport."""
import os
os.environ["COMFY_LOCAL_URL"] = "http://127.0.0.1:8188"

from comfy_mcp import server as comfy_server
comfy_server._apply_startup_instructions()

# Run with streamable-http
comfy_server.mcp.run(transport="streamable-http", host="127.0.0.1", port=8765)
