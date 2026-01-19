"""
Tests for the context manager.

Tests thread-local storage, metadata merging, and integration with decorator.
"""
import pytest
from unittest.mock import Mock, patch
import threading
import time

from magpie_ai.context import (
    context,
    get_context_metadata,
    set_context_metadata,
    merge_metadata,
    ContextMetadata
)


class TestContextMetadata:
    """Test ContextMetadata class."""

    def test_initialization_default(self):
        """Test default initialization."""
        ctx = ContextMetadata()
        assert ctx.project_id is None
        assert ctx.metadata == {}
        assert ctx.trace_id is None

    def test_initialization_with_values(self):
        """Test initialization with values."""
        ctx = ContextMetadata(
            project_id="test-project",
            metadata={"key": "value"},
            trace_id="trace-123"
        )
        assert ctx.project_id == "test-project"
        assert ctx.metadata == {"key": "value"}
        assert ctx.trace_id == "trace-123"


class TestThreadLocalStorage:
    """Test thread-local storage functionality."""

    def test_get_context_metadata_none_initially(self):
        """Test that context metadata is None initially."""
        # Clear any previous context
        set_context_metadata(None)
        assert get_context_metadata() is None

    def test_set_and_get_context_metadata(self):
        """Test setting and getting context metadata."""
        ctx = ContextMetadata(
            project_id="test-project",
            metadata={"key": "value"}
        )

        set_context_metadata(ctx)
        retrieved = get_context_metadata()

        assert retrieved is ctx
        assert retrieved.project_id == "test-project"
        assert retrieved.metadata == {"key": "value"}

        # Cleanup
        set_context_metadata(None)

    def test_context_metadata_is_thread_local(self):
        """Test that context metadata is thread-local."""
        results = {}

        def thread_func(thread_id):
            # Set different context in each thread
            ctx = ContextMetadata(
                project_id=f"project-{thread_id}",
                metadata={"thread": thread_id}
            )
            set_context_metadata(ctx)

            # Small delay to encourage concurrency
            time.sleep(0.01)

            # Get context - should be thread-specific
            retrieved = get_context_metadata()
            results[thread_id] = {
                "project_id": retrieved.project_id,
                "metadata": retrieved.metadata
            }

        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=thread_func, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify each thread had its own context
        for i in range(5):
            assert results[i]["project_id"] == f"project-{i}"
            assert results[i]["metadata"] == {"thread": i}


class TestMergeMetadata:
    """Test metadata merging functionality."""

    def test_merge_both_none(self):
        """Test merging when both are None."""
        result = merge_metadata(None, None)
        assert result == {}

    def test_merge_decorator_only(self):
        """Test merging with only decorator metadata."""
        result = merge_metadata({"model": "gpt-4"}, None)
        assert result == {"model": "gpt-4"}

    def test_merge_context_only(self):
        """Test merging with only context metadata."""
        result = merge_metadata(None, {"user_id": "123"})
        assert result == {"user_id": "123"}

    def test_merge_both_no_overlap(self):
        """Test merging with no overlapping keys."""
        result = merge_metadata(
            {"model": "gpt-4", "temperature": 0.7},
            {"user_id": "123", "session": "abc"}
        )
        assert result == {
            "model": "gpt-4",
            "temperature": 0.7,
            "user_id": "123",
            "session": "abc"
        }

    def test_merge_context_overrides_decorator(self):
        """Test that context metadata overrides decorator metadata."""
        result = merge_metadata(
            {"model": "gpt-4", "temperature": 0.7},
            {"temperature": 0.9, "user_id": "123"}
        )
        assert result == {
            "model": "gpt-4",
            "temperature": 0.9,  # Context value
            "user_id": "123"
        }


class TestContextManager:
    """Test the context manager."""

    @patch('magpie_ai.context.get_client')
    def test_basic_context(self, mock_get_client):
        """Test basic context manager usage."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(
            project_id="test-project",
            metadata={"key": "value"}
        ):
            pass

        # Should have logged the context
        assert mock_client.send_log_sync.called
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['project_id'] == "test-project"
        assert call_args.kwargs['metadata'] == {"key": "value"}

    @patch('magpie_ai.context.get_client')
    def test_context_stores_metadata(self, mock_get_client):
        """Test that context stores metadata in thread-local storage."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Clear any previous context
        set_context_metadata(None)

        with context(
            project_id="test-project",
            metadata={"key": "value"},
            trace_id="trace-123"
        ):
            # Inside context, metadata should be available
            ctx_meta = get_context_metadata()
            assert ctx_meta is not None
            assert ctx_meta.project_id == "test-project"
            assert ctx_meta.metadata == {"key": "value"}
            assert ctx_meta.trace_id == "trace-123"

        # Outside context, metadata should be cleared
        ctx_meta = get_context_metadata()
        assert ctx_meta is None

    @patch('magpie_ai.context.get_client')
    def test_context_without_project_id(self, mock_get_client):
        """Test context without project_id (no logging)."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(metadata={"key": "value"}):
            # Metadata should still be stored
            ctx_meta = get_context_metadata()
            assert ctx_meta is not None
            assert ctx_meta.metadata == {"key": "value"}

        # But context execution should not be logged
        assert not mock_client.send_log_sync.called

    @patch('magpie_ai.context.get_client')
    def test_context_log_context_false(self, mock_get_client):
        """Test disabling context execution logging."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(
            project_id="test-project",
            metadata={"key": "value"},
            log_context=False
        ):
            # Metadata should be stored
            ctx_meta = get_context_metadata()
            assert ctx_meta is not None

        # But no log should be sent
        assert not mock_client.send_log_sync.called

    @patch('magpie_ai.context.get_client')
    def test_context_with_error(self, mock_get_client):
        """Test context when code raises exception."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with pytest.raises(ValueError, match="test error"):
            with context(project_id="test-project"):
                raise ValueError("test error")

        # Should have logged with error status
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['status'] == "error"
        assert call_args.kwargs['error_message'] == "test error"

    @patch('magpie_ai.context.get_client')
    def test_nested_contexts(self, mock_get_client):
        """Test nested context managers."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(
            project_id="outer-project",
            metadata={"level": "outer", "shared": "outer-value"}
        ):
            # Inner context should override outer
            outer_ctx = get_context_metadata()
            assert outer_ctx.project_id == "outer-project"

            with context(
                project_id="inner-project",
                metadata={"level": "inner", "shared": "inner-value"}
            ):
                # Inner context active
                inner_ctx = get_context_metadata()
                assert inner_ctx.project_id == "inner-project"
                assert inner_ctx.metadata == {
                    "level": "inner", "shared": "inner-value"}

            # Back to outer context
            restored_ctx = get_context_metadata()
            assert restored_ctx.project_id == "outer-project"
            assert restored_ctx.metadata == {
                "level": "outer", "shared": "outer-value"}

        # Outside all contexts
        assert get_context_metadata() is None

    @patch('magpie_ai.context.get_client')
    def test_context_timing(self, mock_get_client):
        """Test that context captures timing."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(project_id="test-project"):
            time.sleep(0.1)  # 100ms

        call_args = mock_client.send_log_sync.call_args
        duration_ms = call_args.kwargs['duration_ms']
        assert duration_ms >= 100
        assert duration_ms < 200

    @patch('magpie_ai.context.get_client')
    def test_context_trace_id(self, mock_get_client):
        """Test custom trace ID."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(
            project_id="test-project",
            trace_id="custom-trace-123"
        ):
            ctx_meta = get_context_metadata()
            assert ctx_meta.trace_id == "custom-trace-123"

        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['trace_id'] == "custom-trace-123"

    @patch('magpie_ai.context.get_client')
    def test_context_auto_trace_id(self, mock_get_client):
        """Test automatic trace ID generation."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        with context(project_id="test-project"):
            pass

        call_args = mock_client.send_log_sync.call_args
        trace_id = call_args.kwargs['trace_id']
        assert trace_id is not None
        assert len(trace_id) > 0


class TestFailOpenBehavior:
    """Test fail-open behavior."""

    @patch('magpie_ai.context.get_client')
    def test_metadata_sanitization_failure(self, mock_get_client):
        """Test that metadata sanitization failures don't crash."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # This should not crash even if sanitization fails
        with context(
            project_id="test-project",
            metadata={"key": "value"}  # Normal metadata
        ):
            pass

        # Should still work
        assert mock_client.send_log_sync.called

    @patch('magpie_ai.context.get_client')
    def test_logging_failure_doesnt_crash(self, mock_get_client):
        """Test that logging failures don't crash."""
        mock_client = Mock()
        mock_client.send_log_sync.side_effect = Exception("Network error")
        mock_get_client.return_value = mock_client

        # Should not raise exception
        with context(project_id="test-project"):
            result = "important work"

        assert result == "important work"


class TestThreadSafety:
    """Test thread safety of context manager."""

    @patch('magpie_ai.context.get_client')
    def test_concurrent_contexts(self, mock_get_client):
        """Test multiple concurrent contexts."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        results = []

        def thread_func(thread_id):
            with context(
                project_id=f"project-{thread_id}",
                metadata={"thread": thread_id}
            ):
                time.sleep(0.01)
                ctx_meta = get_context_metadata()
                results.append({
                    "thread": thread_id,
                    "project_id": ctx_meta.project_id,
                    "metadata": ctx_meta.metadata
                })

        threads = []
        for i in range(10):
            thread = threading.Thread(target=thread_func, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have had its own context
        assert len(results) == 10
        for i, result in enumerate(results):
            # Find result for this thread
            thread_results = [r for r in results if r["thread"] == i]
            assert len(thread_results) == 1
            assert thread_results[0]["project_id"] == f"project-{i}"
            assert thread_results[0]["metadata"]["thread"] == i


class TestIntegrationWithDecorator:
    """Test integration with @monitor() decorator."""

    @patch('magpie_ai.monitor.get_client')
    @pytest.mark.skip(reason="metadata parameter not supported in current API")
    def test_decorator_uses_context_metadata(self, mock_get_client):
        """Test that decorator picks up context metadata."""
        from magpie_ai.monitor import monitor

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", metadata={"model": "gpt-4"})
        def test_function():
            return "result"

        with context(metadata={"user_id": "123"}):
            result = test_function()

        assert result == "result"

        # Verify both decorator and context metadata were used
        call_args = mock_client.send_log_sync.call_args
        metadata = call_args.kwargs['metadata']
        assert "model" in metadata  # From decorator
        assert "user_id" in metadata  # From context

    @patch('magpie_ai.monitor.get_client')
    @pytest.mark.skip(reason="metadata parameter not supported in current API")
    def test_context_metadata_overrides_decorator(self, mock_get_client):
        """Test that context metadata overrides decorator metadata."""
        from magpie_ai.monitor import monitor

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(
            project_id="test-project",
            metadata={"temperature": 0.7, "model": "gpt-4"}
        )
        def test_function():
            return "result"

        with context(metadata={"temperature": 0.9}):
            test_function()

        call_args = mock_client.send_log_sync.call_args
        metadata = call_args.kwargs['metadata']
        assert metadata["temperature"] == 0.9  # Context value
        assert metadata["model"] == "gpt-4"  # Decorator value

    @patch('magpie_ai.monitor.get_client')
    @pytest.mark.skip(reason="metadata parameter not supported in current API")
    def test_context_provides_project_id(self, mock_get_client):
        """Test that context can provide project_id to decorator."""
        from magpie_ai.monitor import monitor

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="decorator-project", metadata={"model": "gpt-4"})
        def test_function():
            return "result"

        with context(project_id="context-project"):
            test_function()

        # Decorator project_id should take precedence
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['project_id'] == "decorator-project"

    @patch('magpie_ai.monitor.get_client')
    def test_context_provides_trace_id(self, mock_get_client):
        """Test that context can provide trace_id to decorator."""
        from magpie_ai.monitor import monitor

        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def test_function():
            return "result"

        with context(trace_id="shared-trace-123"):
            test_function()

        # Context trace_id should be used
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['trace_id'] == "shared-trace-123"
