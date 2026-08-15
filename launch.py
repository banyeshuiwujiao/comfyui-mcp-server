"""Launch wrapper for the ComfyUI MCP server.

The embedded ComfyUI Python injects `ComfyUI/` as sys.path[0] via its
sitecustomize, so running server.py directly cannot find sibling modules
(comfyui_client, managers, tools, ...). This wrapper inserts the directory
containing this file at the front of sys.path and then runs server.py as
__main__.
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    server_path = os.path.join(HERE, "server.py")
    runpy.run_path(server_path, run_name="__main__")
