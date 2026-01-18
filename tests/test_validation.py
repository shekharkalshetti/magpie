"""
Tests for client-side metadata validation.

Tests schema caching, validation logic, and integration with client.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from magpie_ai.validation import (
    ValidationResult,
    MetadataSchemaCache,
    validate_metadata,
    sanitize_metadata,
    clear_schema_cache,
    _validate_type,
    _validate_enum
)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.missing_keys == []
        assert result.invalid_types == {}
        assert result.invalid_enum_values == {}
        assert result.unrecognized_keys == []

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = ValidationResult(
            is_valid=False,
            missing_keys=["user_id"],
            invalid_types={"temperature": "Expected int, got str"},
            invalid_enum_values={"env": "Not in allowed values"},
            unrecognized_keys=["unknown"]
        )
        assert result.is_valid is False
        assert result.missing_keys == ["user_id"]
        assert result.invalid_types == {"temperature": "Expected int, got str"}
        assert result.invalid_enum_values == {"env": "Not in allowed values"}
        assert result.unrecognized_keys == ["unknown"]

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ValidationResult(
            is_valid=False,
            missing_keys=["user_id"],
            invalid_types={"temperature": "Expected int, got str"}
        )

        dict_result = result.to_dict()
        assert dict_result["is_valid"] is False
        assert dict_result["missing_keys"] == ["user_id"]
        assert dict_result["invalid_types"] == {
            "temperature": "Expected int, got str"}
        # Empty lists/dicts should not be included
        assert "invalid_enum_values" not in dict_result
        assert "unrecognized_keys" not in dict_result


class TestMetadataSchemaCache:
    """Test MetadataSchemaCache class."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = MetadataSchemaCache(ttl_seconds=60)
        assert cache.ttl_seconds == 60
        assert cache._cache == {}
        assert cache._cache_times == {}

    def test_cache_miss_and_fetch(self):
        """Test cache miss triggers fetch."""
        cache = MetadataSchemaCache()

        # Mock the fetch method
        mock_schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None}
        }

        with patch.object(cache, '_fetch_schema', return_value=mock_schema):
            schema = cache.get_schema("project-1", "http://backend", "api-key")

            assert schema == mock_schema
            assert "project-1" in cache._cache
            assert "project-1" in cache._cache_times

    def test_cache_hit(self):
        """Test cache hit returns cached value."""
        cache = MetadataSchemaCache()

        # Pre-populate cache
        mock_schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None}
        }
        cache._cache["project-1"] = mock_schema
        cache._cache_times["project-1"] = datetime.utcnow()

        # Get from cache (no fetch should occur)
        with patch.object(cache, '_fetch_schema') as mock_fetch:
            schema = cache.get_schema("project-1", "http://backend", "api-key")

            assert schema == mock_schema
            mock_fetch.assert_not_called()

    def test_cache_expiry(self):
        """Test cache expiry triggers re-fetch."""
        cache = MetadataSchemaCache(ttl_seconds=60)

        # Pre-populate cache with old timestamp
        old_schema = {"old": "schema"}
        new_schema = {"new": "schema"}
        cache._cache["project-1"] = old_schema
        cache._cache_times["project-1"] = datetime.utcnow() - \
            timedelta(seconds=120)

        # Get from cache (should re-fetch)
        with patch.object(cache, '_fetch_schema', return_value=new_schema):
            schema = cache.get_schema("project-1", "http://backend", "api-key")

            assert schema == new_schema
            assert cache._cache["project-1"] == new_schema

    def test_fetch_failure_returns_none(self):
        """Test fetch failure returns None."""
        cache = MetadataSchemaCache()

        with patch.object(cache, '_fetch_schema', side_effect=Exception("Network error")):
            schema = cache.get_schema("project-1", "http://backend", "api-key")

            assert schema is None

    def test_clear_cache(self):
        """Test clearing cache."""
        cache = MetadataSchemaCache()
        cache._cache["project-1"] = {"model": "schema"}
        cache._cache_times["project-1"] = datetime.utcnow()

        cache.clear()

        assert cache._cache == {}
        assert cache._cache_times == {}


class TestTypeValidation:
    """Test type validation functions."""

    def test_validate_string_type(self):
        """Test string type validation."""
        assert _validate_type("hello", "string") is None
        assert _validate_type(123, "string") is not None
        assert _validate_type(True, "string") is not None

    def test_validate_int_type(self):
        """Test int type validation."""
        assert _validate_type(123, "int") is None
        assert _validate_type("123", "int") is not None
        # Bools are not ints for our purposes
        assert _validate_type(True, "int") is not None

    def test_validate_bool_type(self):
        """Test bool type validation."""
        assert _validate_type(True, "bool") is None
        assert _validate_type(False, "bool") is None
        assert _validate_type(1, "bool") is not None
        assert _validate_type("true", "bool") is not None

    def test_validate_enum_type(self):
        """Test enum type validation."""
        assert _validate_type("value", "enum") is None
        assert _validate_type(123, "enum") is not None

    def test_validate_none_values(self):
        """Test that None is allowed for all types."""
        assert _validate_type(None, "string") is None
        assert _validate_type(None, "int") is None
        assert _validate_type(None, "bool") is None
        assert _validate_type(None, "enum") is None


class TestEnumValidation:
    """Test enum validation function."""

    def test_validate_allowed_value(self):
        """Test allowed enum value."""
        assert _validate_enum("dev", ["dev", "staging", "prod"]) is None

    def test_validate_disallowed_value(self):
        """Test disallowed enum value."""
        error = _validate_enum("local", ["dev", "staging", "prod"])
        assert error is not None
        assert "local" in error
        assert "dev" in error

    def test_validate_none_value(self):
        """Test None is allowed."""
        assert _validate_enum(None, ["dev", "staging", "prod"]) is None

    def test_validate_no_allowed_values(self):
        """Test when no allowed values specified."""
        assert _validate_enum("anything", None) is None
        assert _validate_enum("anything", []) is None

    def test_validate_non_string_value(self):
        """Test non-string enum value."""
        error = _validate_enum(123, ["dev", "staging", "prod"])
        assert error is not None
        assert "string" in error.lower()


class TestMetadataValidation:
    """Test metadata validation function."""

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_with_valid_metadata(self, mock_cache):
        """Test validation with valid metadata."""
        schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None},
            "temperature": {"value_type": "int", "required": False, "enum_values": None}
        }
        mock_cache.get_schema.return_value = schema

        metadata = {"model": "gpt-4", "temperature": 7}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        assert result.is_valid is True
        assert result.missing_keys == []
        assert result.invalid_types == {}

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_with_missing_required_keys(self, mock_cache):
        """Test validation with missing required keys."""
        schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None},
            "user_id": {"value_type": "string", "required": True, "enum_values": None}
        }
        mock_cache.get_schema.return_value = schema

        metadata = {"model": "gpt-4"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        assert result.is_valid is False
        assert "user_id" in result.missing_keys

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_with_invalid_types(self, mock_cache):
        """Test validation with invalid types."""
        schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None},
            "temperature": {"value_type": "int", "required": False, "enum_values": None}
        }
        mock_cache.get_schema.return_value = schema

        metadata = {"model": 123, "temperature": "high"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        assert result.is_valid is False
        assert "model" in result.invalid_types
        assert "temperature" in result.invalid_types

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_with_invalid_enum_values(self, mock_cache):
        """Test validation with invalid enum values."""
        schema = {
            "environment": {
                "value_type": "enum",
                "required": True,
                "enum_values": ["dev", "staging", "prod"]
            }
        }
        mock_cache.get_schema.return_value = schema

        metadata = {"environment": "local"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        assert result.is_valid is False
        assert "environment" in result.invalid_enum_values

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_with_unrecognized_keys(self, mock_cache):
        """Test validation with unrecognized keys."""
        schema = {
            "model": {"value_type": "string", "required": True, "enum_values": None}
        }
        mock_cache.get_schema.return_value = schema

        metadata = {"model": "gpt-4", "unknown_field": "value"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        assert result.is_valid is True  # Unrecognized keys don't invalidate
        assert "unknown_field" in result.unrecognized_keys

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_without_schema(self, mock_cache):
        """Test validation when schema is unavailable."""
        mock_cache.get_schema.return_value = None

        metadata = {"model": "gpt-4"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        # Should return valid when schema unavailable (fail-open)
        assert result.is_valid is True

    @patch('magpie_ai.validation._schema_cache')
    def test_validation_exception_handling(self, mock_cache):
        """Test validation handles exceptions gracefully."""
        mock_cache.get_schema.side_effect = Exception("Network error")

        metadata = {"model": "gpt-4"}
        result = validate_metadata(
            metadata, "project-1", "http://backend", "api-key")

        # Should return valid on exception (fail-open)
        assert result.is_valid is True


class TestSanitizeMetadata:
    """Test metadata sanitization."""

    def test_sanitize_primitives(self):
        """Test sanitizing primitive types."""
        metadata = {
            "string": "value",
            "int": 123,
            "float": 3.14,
            "bool": True,
            "none": None
        }

        result = sanitize_metadata(metadata)
        assert result == metadata

    def test_sanitize_collections(self):
        """Test sanitizing lists and dicts."""
        metadata = {
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }

        result = sanitize_metadata(metadata)
        assert result == metadata

    def test_sanitize_non_serializable(self):
        """Test sanitizing non-serializable values."""
        class CustomClass:
            def __str__(self):
                return "custom"

        metadata = {
            "callable": lambda x: x,
            "object": CustomClass(),
            "datetime": datetime(2024, 1, 1)
        }

        result = sanitize_metadata(metadata)
        assert isinstance(result["callable"], str)
        assert result["object"] == "custom"
        assert isinstance(result["datetime"], str)


class TestGlobalFunctions:
    """Test global functions."""

    def test_clear_schema_cache(self):
        """Test clearing global schema cache."""
        from magpie_ai.validation import _schema_cache

        # Add some data to cache
        _schema_cache._cache["project-1"] = {"model": "schema"}
        _schema_cache._cache_times["project-1"] = datetime.utcnow()

        # Clear cache
        clear_schema_cache()

        # Verify cache is empty
        assert _schema_cache._cache == {}
        assert _schema_cache._cache_times == {}
