"""
HTTP client for communicating with Magpie backend.

Handles async POST requests to log execution data.
"""

import httpx
from typing import Dict, Any, Optional
from datetime import datetime

from magpie_ai.config import get_config
from magpie_ai.validation import validate_metadata, sanitize_metadata


class MagpieClient:
    """Client for Magpie backend API."""

    def __init__(self):
        self.config = get_config()

    async def send_log(
        self,
        project_id: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        duration_ms: Optional[int] = None,
        status: Optional[str] = None,
        error_message: Optional[str] = None,
        function_name: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> bool:
        """
        Send execution log to backend.

        Returns True if successful, False otherwise.
        Implements fail-open behavior - won't raise exceptions.
        All network calls are fail-open and will not crash the application.

        Performs client-side metadata validation before sending.
        """
        if not self.config.enabled:
            return True

        if not self.config.is_configured():
            if not self.config.fail_open:
                raise RuntimeError(
                    "Magpie SDK not configured. Set MAGPIE_API_KEY and MAGPIE_BACKEND_URL."
                )
            return False

        try:
            # Sanitize metadata
            if metadata:
                metadata = sanitize_metadata(metadata)

            # Perform client-side validation (best-effort, never blocks)
            validation_result = None
            if metadata and project_id:
                validation_result = validate_metadata(
                    metadata=metadata,
                    project_id=project_id,
                    backend_url=self.config.backend_url,
                    api_key=self.config.api_key,
                )

            # Prepare payload
            payload = {
                "input": input,
                "output": output,
                "metadata": metadata,
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "duration_ms": duration_ms,
                "status": status,
                "error_message": error_message,
                "function_name": function_name,
                "trace_id": trace_id,
            }

            # Include validation result if available
            if validation_result:
                payload["metadata_valid"] = validation_result.is_valid
                if not validation_result.is_valid:
                    payload["validation_errors"] = validation_result.to_dict()

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Send request
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    # Note: trailing slash required by FastAPI
                    f"{self.config.backend_url}/api/v1/logs/",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )

                response.raise_for_status()
                return True

        except httpx.HTTPStatusError as e:
            # HTTP error (4xx, 5xx)
            if self.config.fail_open:
                print(f"[Magpie] Warning: HTTP error {e.response.status_code}: {e}")
                return False
            else:
                raise
        except httpx.TimeoutException as e:
            # Request timeout
            if self.config.fail_open:
                print(f"[Magpie] Warning: Request timeout: {e}")
                return False
            else:
                raise
        except Exception as e:
            # Any other error (network, etc.)
            if self.config.fail_open:
                print(f"[Magpie] Warning: Failed to send log: {e}")
                return False
            else:
                raise

    def send_log_sync(self, **kwargs) -> Optional[str]:
        """
        Synchronous HTTP request - no asyncio, no event loops.
        This is the SOLID solution that works even during interpreter shutdown.

        Returns the log ID from the response, or None if send failed.
        """
        if not self.config.enabled:
            return None

        if not self.config.is_configured():
            if not self.config.fail_open:
                raise RuntimeError(
                    "Magpie SDK not configured. Set MAGPIE_API_KEY and MAGPIE_BACKEND_URL."
                )
            return None

        try:
            # Extract parameters
            project_id = kwargs.get("project_id")
            input_text = kwargs.get("input")
            output_text = kwargs.get("output")
            custom = kwargs.get("custom")
            started_at = kwargs.get("started_at")
            completed_at = kwargs.get("completed_at")
            total_latency_ms = kwargs.get("total_latency_ms")
            status = kwargs.get("status")
            error_message = kwargs.get("error_message")
            function_name = kwargs.get("function_name")
            trace_id = kwargs.get("trace_id")

            # New metrics parameters
            input_tokens = kwargs.get("input_tokens")
            output_tokens = kwargs.get("output_tokens")
            context_utilization = kwargs.get("context_utilization", 0.0)
            input_cost = kwargs.get("input_cost", 0.0)
            output_cost = kwargs.get("output_cost", 0.0)
            pii_info = kwargs.get("pii_info")
            moderation_info = kwargs.get("moderation_info")

            # Calculate total_tokens if we have both
            total_tokens = None
            if input_tokens and output_tokens:
                total_tokens = input_tokens + output_tokens

            # Calculate total_cost
            total_cost = (input_cost or 0.0) + (output_cost or 0.0)

            # Prepare payload with new schema
            payload = {
                "project_id": project_id,
                "trace_id": trace_id,
                "input": input_text,
                "output": output_text,
                "custom": custom,
                "started_at": started_at.isoformat() if started_at else None,
                "completed_at": completed_at.isoformat() if completed_at else None,
                "total_latency_ms": total_latency_ms,
                "status": status,
                "error_message": error_message,
                "function_name": function_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "context_utilization": context_utilization,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "pii_detection": pii_info,
                "content_moderation": moderation_info,
            }

            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}

            # Send synchronous HTTP request using httpx
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self.config.backend_url}/api/v1/logs/",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )

                response.raise_for_status()

                # Try to extract log ID from response
                try:
                    response_data = response.json()
                    log_id = response_data.get("id")
                    if isinstance(log_id, str):
                        return log_id
                    return None
                except Exception:
                    return None

        except httpx.HTTPStatusError:
            # HTTP error (4xx, 5xx)
            if self.config.fail_open:
                return None
            else:
                raise
        except httpx.TimeoutException:
            # Request timeout
            if self.config.fail_open:
                return None
            else:
                raise
        except Exception:
            # Any other error - fail silently during shutdown
            if self.config.fail_open:
                return None
            else:
                raise


# Global client instance
_client: Optional[MagpieClient] = None


def get_client() -> MagpieClient:
    """Get global client instance."""
    global _client
    if _client is None:
        _client = MagpieClient()
    return _client
