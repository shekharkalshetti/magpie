"""
Configuration for Triton SDK.

Loads settings from environment variables.
"""
import os
from typing import Optional


class TritonConfig:
    """SDK configuration."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        backend_url: Optional[str] = None,
        enabled: bool = True,
        timeout: int = 5,
        fail_open: bool = True
    ):
        """
        Initialize Triton SDK configuration.

        Args:
            api_key: Triton API key (defaults to MAGPIE_API_KEY env var)
            backend_url: Backend URL (defaults to MAGPIE_BACKEND_URL or http://localhost:8000)
            enabled: Whether monitoring is enabled (defaults to MAGPIE_ENABLED or True)
            timeout: Request timeout in seconds
            fail_open: If True, failures won't crash the app
        """
        self.api_key = api_key or os.getenv("MAGPIE_API_KEY")
        self.backend_url = backend_url or os.getenv(
            "MAGPIE_BACKEND_URL", "http://localhost:8000")
        self.enabled = enabled and os.getenv(
            "MAGPIE_ENABLED", "true").lower() != "false"
        self.timeout = timeout
        self.fail_open = fail_open

        # Ensure backend URL doesn't end with slash
        if self.backend_url and self.backend_url.endswith("/"):
            self.backend_url = self.backend_url[:-1]

    def is_configured(self) -> bool:
        """Check if SDK is properly configured."""
        return bool(self.api_key and self.backend_url)


# Global config instance
_config: Optional[TritonConfig] = None


def get_config() -> TritonConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = TritonConfig()
    return _config


def configure(
    api_key: Optional[str] = None,
    backend_url: Optional[str] = None,
    enabled: bool = True,
    timeout: int = 5,
    fail_open: bool = True
):
    """
    Configure the Triton SDK.

    Call this at application startup to set configuration.

    Example:
        import magpie_ai
        magpie_ai.configure(
            api_key="tr_your_api_key",
            backend_url="https://api.magpie_ai.dev"
        )
    """
    global _config
    _config = TritonConfig(
        api_key=api_key,
        backend_url=backend_url,
        enabled=enabled,
        timeout=timeout,
        fail_open=fail_open
    )
