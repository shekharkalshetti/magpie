"""
Default configuration for the Magpie SDK.

Single source of truth for LLM URL and model defaults.
Reads from environment variables, falling back to local dev values.
Set VLLM_URL and VLLM_MODEL in your environment or .env to override.
"""

import os

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:1234")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3-1.7b")
