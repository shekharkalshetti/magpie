"""
Metadata validation utilities.

Client-side validation of metadata against project schema.
Implements caching and best-effort validation that never blocks execution.
"""

import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime


class ValidationResult:
    """Result of metadata validation."""

    def __init__(
        self,
        is_valid: bool = True,
        missing_keys: Optional[List[str]] = None,
        invalid_types: Optional[Dict[str, str]] = None,
        invalid_enum_values: Optional[Dict[str, str]] = None,
        unrecognized_keys: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.missing_keys: List[str] = missing_keys if missing_keys is not None else []
        self.invalid_types: Dict[str, str] = invalid_types if invalid_types is not None else {}
        self.invalid_enum_values: Dict[str, str] = (
            invalid_enum_values if invalid_enum_values is not None else {}
        )
        self.unrecognized_keys: List[str] = (
            unrecognized_keys if unrecognized_keys is not None else []
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result: Dict[str, Any] = {"is_valid": self.is_valid}
        if self.missing_keys:
            result["missing_keys"] = self.missing_keys
        if self.invalid_types:
            result["invalid_types"] = self.invalid_types
        if self.invalid_enum_values:
            result["invalid_enum_values"] = self.invalid_enum_values
        if self.unrecognized_keys:
            result["unrecognized_keys"] = self.unrecognized_keys
        return result


class MetadataSchemaCache:
    """
    Cache for metadata schemas per project.

    Fetches schema from backend and caches it with TTL.
    Never blocks execution - all failures are silent.
    """

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize schema cache.

        Args:
            ttl_seconds: Time-to-live for cached schemas (default: 5 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_times: Dict[str, datetime] = {}
        self.ttl_seconds = ttl_seconds

    def get_schema(
        self, project_id: str, backend_url: str, api_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata schema for a project.

        Returns cached schema if available and fresh, otherwise fetches from backend.
        Fails silently and returns None on any error.

        Args:
            project_id: Project ID
            backend_url: Backend URL
            api_key: API key for authentication

        Returns:
            Schema dictionary or None if unavailable
        """
        # Check cache
        if project_id in self._cache:
            cache_time = self._cache_times.get(project_id)
            if cache_time and (datetime.utcnow() - cache_time).total_seconds() < self.ttl_seconds:
                return self._cache[project_id]

        # Fetch from backend
        try:
            schema = self._fetch_schema(project_id, backend_url, api_key)
            if schema:
                self._cache[project_id] = schema
                self._cache_times[project_id] = datetime.utcnow()
            return schema
        except Exception:
            # Fail silently - never block execution
            return None

    def _fetch_schema(
        self, project_id: str, backend_url: str, api_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch metadata schema from backend.

        Args:
            project_id: Project ID
            backend_url: Backend URL
            api_key: API key for authentication

        Returns:
            Schema dictionary or None on error
        """
        try:
            # Make synchronous request
            with httpx.Client(timeout=2.0) as client:
                response = client.get(
                    f"{backend_url}/api/v1/projects/{project_id}/metadata-keys",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()

                # Convert list of metadata keys to schema dict
                metadata_keys = response.json()
                schema = {}
                for key_config in metadata_keys:
                    schema[key_config["key"]] = {
                        "value_type": key_config["value_type"],
                        "required": key_config["required"],
                        "enum_values": key_config.get("enum_values"),
                    }

                return schema
        except Exception:
            # Fail silently
            return None

    def clear(self):
        """Clear all cached schemas."""
        self._cache.clear()
        self._cache_times.clear()


# Global schema cache instance
_schema_cache = MetadataSchemaCache()


def validate_metadata(
    metadata: Dict[str, Any], project_id: str, backend_url: str, api_key: str
) -> ValidationResult:
    """
    Validate metadata against project schema.

    Performs best-effort validation:
    - Fetches and caches schema from backend
    - Validates required keys, types, and enum values
    - Never blocks execution (fails silently)
    - Returns validation result for logging

    Args:
        metadata: Metadata dictionary to validate
        project_id: Project ID
        backend_url: Backend URL
        api_key: API key for authentication

    Returns:
        ValidationResult with detailed validation information
    """
    try:
        # Get schema (from cache or backend)
        schema = _schema_cache.get_schema(project_id, backend_url, api_key)

        # If schema unavailable, skip validation
        if not schema:
            return ValidationResult(is_valid=True)

        # Perform validation
        missing_keys: List[str] = []
        invalid_types: Dict[str, str] = {}
        invalid_enum_values: Dict[str, str] = {}
        unrecognized_keys: List[str] = []

        # Check for missing required keys
        for key, config in schema.items():
            if config["required"] and key not in metadata:
                missing_keys.append(key)

        # Validate provided keys
        for key, value in metadata.items():
            # Skip system metadata keys (not part of user schema)
            if key in ["pii_detection", "pii_detection_error"]:
                continue

            # Check if key is recognized
            if key not in schema:
                unrecognized_keys.append(key)
                continue

            config = schema[key]
            value_type = config["value_type"]

            # Validate type
            type_error = _validate_type(value, value_type)
            if type_error:
                invalid_types[key] = type_error

            # Validate enum values
            if value_type == "enum" and value is not None:
                enum_error = _validate_enum(value, config.get("enum_values"))
                if enum_error:
                    invalid_enum_values[key] = enum_error

        # Determine overall validity
        is_valid = not (missing_keys or invalid_types or invalid_enum_values)

        return ValidationResult(
            is_valid=is_valid,
            missing_keys=missing_keys,
            invalid_types=invalid_types,
            invalid_enum_values=invalid_enum_values,
            unrecognized_keys=unrecognized_keys,
        )

    except Exception:
        # If validation fails, return valid result (fail-open)
        return ValidationResult(is_valid=True)


def _validate_type(value: Any, expected_type: str) -> Optional[str]:
    """
    Validate that a value matches the expected type.

    Args:
        value: The value to validate
        expected_type: Expected type ("string", "int", "bool", "enum")

    Returns:
        Error message if invalid, None if valid
    """
    if value is None:
        return None  # Allow None for optional fields

    if expected_type == "string":
        if not isinstance(value, str):
            return f"Expected string, got {type(value).__name__}"

    elif expected_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            # Note: isinstance(True, int) is True in Python, so we exclude bools
            return f"Expected int, got {type(value).__name__}"

    elif expected_type == "bool":
        if not isinstance(value, bool):
            return f"Expected bool, got {type(value).__name__}"

    elif expected_type == "enum":
        if not isinstance(value, str):
            return f"Expected string (enum), got {type(value).__name__}"

    return None


def _validate_enum(value: Any, allowed_values: Optional[List[str]]) -> Optional[str]:
    """
    Validate that an enum value is in the allowed list.

    Args:
        value: The value to validate
        allowed_values: List of allowed enum values

    Returns:
        Error message if invalid, None if valid
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return f"Enum value must be a string, got {type(value).__name__}"

    if not allowed_values:
        return None  # No validation if no allowed values specified

    if value not in allowed_values:
        return f"Value '{value}' not in allowed values: {allowed_values}"

    return None


def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metadata for safe transmission.

    Converts non-serializable values to strings.

    Args:
        metadata: Raw metadata dict

    Returns:
        Sanitized metadata dict
    """
    sanitized = {}

    for key, value in metadata.items():
        # Keep primitives as-is
        if isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value  # type: ignore[assignment]
        # Keep lists and dicts (assume they're serializable)
        elif isinstance(value, (list, dict)):
            sanitized[key] = value  # type: ignore[assignment]
        # Convert everything else to string
        else:
            sanitized[key] = str(value)

    return sanitized


def clear_schema_cache():
    """Clear the global schema cache. Useful for testing."""
    _schema_cache.clear()
