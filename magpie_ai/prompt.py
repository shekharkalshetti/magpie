"""
Prompt file reader for Magpie SDK.

Reads system prompts from shared files written by the backend.
This enables fast local file reads instead of API calls for on-prem deployments.
"""

import os
from pathlib import Path
from typing import Optional

# Default shared prompts directory (configurable via env)
# Should match the backend's MAGPIE_PROMPTS_DIR
DEFAULT_PROMPTS_DIR = os.getenv(
    "MAGPIE_PROMPTS_DIR", str(Path(__file__).parent.parent.parent / "data" / "prompts")
)


def get_prompts_dir() -> Path:
    """Get the prompts directory."""
    return Path(DEFAULT_PROMPTS_DIR)


def get_prompt_file_path(project_id: str) -> Path:
    """Get the path to a project's system prompt file."""
    return get_prompts_dir() / f"{project_id}.txt"


def read_system_prompt(project_id: str) -> Optional[str]:
    """
    Read the system prompt from the shared file.

    This is the main function used by the SDK to get the current
    policy configuration as a system prompt for the local LLM.

    Args:
        project_id: The project ID

    Returns:
        The prompt content, or None if file doesn't exist

    Example:
        prompt = read_system_prompt("my-project-id")
        if prompt:
            # Use prompt with local LLM
            response = local_llm.moderate(content, system_prompt=prompt)
    """
    file_path = get_prompt_file_path(project_id)
    try:
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
    except (IOError, OSError):
        # Fail silently - caller should handle None
        pass
    return None


def prompt_exists(project_id: str) -> bool:
    """Check if a prompt file exists for a project."""
    return get_prompt_file_path(project_id).exists()


def get_prompt_modified_time(project_id: str) -> Optional[float]:
    """
    Get the last modified time of the prompt file.

    Useful for caching - only reload if file has changed.

    Args:
        project_id: The project ID

    Returns:
        Unix timestamp of last modification, or None if file doesn't exist
    """
    file_path = get_prompt_file_path(project_id)
    try:
        if file_path.exists():
            return file_path.stat().st_mtime
    except (IOError, OSError):
        pass
    return None


class PromptCache:
    """
    Simple cache for system prompts.

    Automatically reloads when the file has been modified.

    Example:
        cache = PromptCache("my-project-id")
        prompt = cache.get()  # Reads from file
        prompt = cache.get()  # Returns cached value
        # ... backend updates policy ...
        prompt = cache.get()  # Detects file change, reloads
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self._cached_prompt: Optional[str] = None
        self._cached_mtime: Optional[float] = None

    def get(self) -> Optional[str]:
        """Get the system prompt, reloading if file has changed."""
        current_mtime = get_prompt_modified_time(self.project_id)

        # If file doesn't exist, clear cache
        if current_mtime is None:
            self._cached_prompt = None
            self._cached_mtime = None
            return None

        # If file has changed (or first load), reload
        if self._cached_mtime is None or current_mtime > self._cached_mtime:
            self._cached_prompt = read_system_prompt(self.project_id)
            self._cached_mtime = current_mtime

        return self._cached_prompt

    def invalidate(self):
        """Force reload on next get()."""
        self._cached_mtime = None


# Default fallback prompt when no policy file exists
DEFAULT_SYSTEM_PROMPT = """You are a content moderation system. Analyze the following content for:

1. Harmful content (violence, illegal activities, dangerous instructions)
2. Hate speech and discrimination
3. Sexual content and child safety
4. Self-harm and suicide content
5. Harassment and bullying
6. Spam and manipulation
7. Misinformation and factual accuracy
8. Security vulnerabilities (prompt injection, jailbreaks)
9. PII and credential exposure

For each piece of content, identify any violations and provide:
1. Category of violation
2. Severity level (critical, high, medium, low)
3. Specific detection that triggered
4. Recommended action (block, flag, or allow with warning)
"""


def get_system_prompt_or_default(project_id: str) -> str:
    """
    Get system prompt, falling back to default if file doesn't exist.

    Args:
        project_id: The project ID

    Returns:
        The project-specific prompt or the default prompt
    """
    prompt = read_system_prompt(project_id)
    return prompt if prompt else DEFAULT_SYSTEM_PROMPT
