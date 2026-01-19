"""
Magpie SDK - LLM Middleware for monitoring and metadata tracking.

Provides decorator and context manager for capturing LLM execution traces.
"""

__version__ = "0.2.6"

from magpie_ai.monitor import monitor
from magpie_ai.context import context
from magpie_ai.validation import validate_metadata, clear_schema_cache, ValidationResult
from magpie_ai.content_moderation import ContentModerationError

__all__ = [
    "monitor",
    "context",
    "validate_metadata",
    "clear_schema_cache",
    "ValidationResult",
    "ContentModerationError",
]
