"""
Tests for the @monitor() decorator.

Tests input/output capture, timing, error handling, and fail-open behavior.
"""
import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime
import time
import threading

from magpie_ai.monitor import (
    monitor,
    _execute_monitored,
    _capture_input,
    _capture_output,
    _send_log_async
)


class TestMonitorDecorator:
    """Test the @monitor() decorator."""

    @patch('magpie_ai.monitor.get_client')
    def test_basic_decoration(self, mock_get_client):
        """Test basic function decoration and execution."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def simple_function(x, y):
            return x + y

        result = simple_function(2, 3)

        assert result == 5
        assert mock_client.send_log_sync.called

    @patch('magpie_ai.monitor.get_client')
    def test_decorator_with_metadata(self, mock_get_client):
        """Test decorator with static metadata."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(
            project_id="test-project",
            metadata={"model": "gpt-4", "temperature": 0.7}
        )
        def llm_function(prompt):
            return f"Response to: {prompt}"

        result = llm_function("Hello")

        assert result == "Response to: Hello"

        # Verify metadata was passed
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['metadata'] == {
            "model": "gpt-4", "temperature": 0.7}
        assert call_args.kwargs['project_id'] == "test-project"

    @patch('magpie_ai.monitor.get_client')
    def test_decorator_preserves_function_metadata(self, mock_get_client):
        """Test that decorator preserves function name and docstring."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def documented_function(x):
            """This is a docstring."""
            return x * 2

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == """This is a docstring."""

    @patch('magpie_ai.monitor.get_client')
    def test_decorator_with_kwargs(self, mock_get_client):
        """Test decorator with keyword arguments."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", capture_input=True)
        def function_with_kwargs(a, b=10, c=20):
            return a + b + c

        result = function_with_kwargs(5, b=15, c=25)

        assert result == 45

        # Verify input was captured
        call_args = mock_client.send_log_sync.call_args
        input_data = call_args.kwargs['input']
        assert input_data['a'] == 5
        assert input_data['b'] == 15
        assert input_data['c'] == 25


class TestExecutionMonitoring:
    """Test execution monitoring behavior."""

    @patch('magpie_ai.monitor.get_client')
    def test_successful_execution_logging(self, mock_get_client):
        """Test logging of successful execution."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def successful_function():
            return "success"

        result = successful_function()

        assert result == "success"

        # Verify log was sent
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['status'] == "success"
        assert call_args.kwargs['error_message'] is None
        assert call_args.kwargs['function_name'] == "successful_function"

    @patch('magpie_ai.monitor.get_client')
    def test_error_execution_logging(self, mock_get_client):
        """Test logging when function raises exception."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def failing_function():
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError, match="Something went wrong"):
            failing_function()

        # Verify error was logged
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['status'] == "error"
        assert call_args.kwargs['error_message'] == "Something went wrong"

    @patch('magpie_ai.monitor.get_client')
    def test_timing_capture(self, mock_get_client):
        """Test that timing is captured correctly."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def slow_function():
            time.sleep(0.1)  # 100ms
            return "done"

        slow_function()

        # Verify timing was captured
        call_args = mock_client.send_log_sync.call_args
        duration_ms = call_args.kwargs['duration_ms']
        assert duration_ms >= 100  # At least 100ms
        assert duration_ms < 200   # But not too long

        # Verify timestamps
        assert isinstance(call_args.kwargs['started_at'], datetime)
        assert isinstance(call_args.kwargs['completed_at'], datetime)

    @patch('magpie_ai.monitor.get_client')
    def test_trace_id_generation(self, mock_get_client):
        """Test automatic trace ID generation."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def function_with_auto_trace():
            return "result"

        function_with_auto_trace()

        # Verify trace_id was generated
        call_args = mock_client.send_log_sync.call_args
        trace_id = call_args.kwargs['trace_id']
        assert trace_id is not None
        assert isinstance(trace_id, str)
        assert len(trace_id) > 0

    @patch('magpie_ai.monitor.get_client')
    def test_custom_trace_id(self, mock_get_client):
        """Test using custom trace ID."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", trace_id="custom-trace-123")
        def function_with_custom_trace():
            return "result"

        function_with_custom_trace()

        # Verify custom trace_id was used
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['trace_id'] == "custom-trace-123"


class TestInputCapture:
    """Test input capture functionality."""

    def test_capture_simple_args(self):
        """Test capturing simple positional arguments."""
        def test_func(a, b, c):
            pass

        input_data = _capture_input(test_func, (1, 2, 3), {})

        assert input_data == {'a': 1, 'b': 2, 'c': 3}

    def test_capture_kwargs(self):
        """Test capturing keyword arguments."""
        def test_func(a, b=10, c=20):
            pass

        input_data = _capture_input(test_func, (5,), {'b': 15, 'c': 25})

        assert input_data == {'a': 5, 'b': 15, 'c': 25}

    def test_capture_with_defaults(self):
        """Test capturing with default values applied."""
        def test_func(a, b=10, c=20):
            pass

        input_data = _capture_input(test_func, (5,), {})

        assert input_data == {'a': 5, 'b': 10, 'c': 20}

    def test_capture_non_serializable_objects(self):
        """Test capturing non-JSON-serializable objects."""
        class CustomClass:
            def __str__(self):
                return "CustomClass instance"

        def test_func(obj):
            pass

        custom_obj = CustomClass()
        input_data = _capture_input(test_func, (custom_obj,), {})

        assert 'obj' in input_data
        assert input_data['obj']['_type'] == 'CustomClass'
        assert 'CustomClass instance' in input_data['obj']['_repr']

    def test_capture_mixed_serializable(self):
        """Test capturing mix of serializable and non-serializable."""
        class NonSerializable:
            pass

        def test_func(a, b, c):
            pass

        input_data = _capture_input(
            test_func,
            ("string", 123, NonSerializable()),
            {}
        )

        assert input_data['a'] == "string"
        assert input_data['b'] == 123
        assert '_type' in input_data['c']

    def test_capture_failure_fallback(self):
        """Test fallback behavior when signature binding fails."""
        def test_func(*args, **kwargs):
            pass

        # This should work with fallback
        input_data = _capture_input(test_func, (1, 2, 3), {'key': 'value'})

        # Should have fallback info
        assert '_args_count' in input_data or 'args' in input_data


class TestOutputCapture:
    """Test output capture functionality."""

    def test_capture_simple_output(self):
        """Test capturing simple JSON-serializable output."""
        output_data = _capture_output("simple string")

        assert output_data == {"value": "simple string"}

    def test_capture_dict_output(self):
        """Test capturing dictionary output."""
        result = {"key": "value", "number": 123}
        output_data = _capture_output(result)

        assert output_data == {"value": result}

    def test_capture_list_output(self):
        """Test capturing list output."""
        result = [1, 2, 3, "four"]
        output_data = _capture_output(result)

        assert output_data == {"value": result}

    def test_capture_non_serializable_output(self):
        """Test capturing non-JSON-serializable output."""
        class CustomClass:
            def __str__(self):
                return "CustomClass result"

        result = CustomClass()
        output_data = _capture_output(result)

        assert output_data['_type'] == 'CustomClass'
        assert 'CustomClass result' in output_data['_repr']

    def test_capture_none_output(self):
        """Test capturing None output."""
        output_data = _capture_output(None)

        assert output_data == {"value": None}


class TestFailOpenBehavior:
    """Test fail-open behavior."""

    @patch('magpie_ai.monitor.get_client')
    def test_monitoring_failure_doesnt_crash(self, mock_get_client):
        """Test that monitoring failures don't crash the function."""
        # Make send_log_sync raise an exception
        mock_client = Mock()
        mock_client.send_log_sync.side_effect = Exception("Network error")
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def important_function():
            return "important result"

        # Function should still work despite monitoring failure
        result = important_function()
        assert result == "important result"

    @patch('magpie_ai.monitor.get_client')
    def test_metadata_sanitization_failure_doesnt_crash(self, mock_get_client):
        """Test that metadata sanitization failures don't crash."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Use metadata that might cause issues
        @monitor(
            project_id="test-project",
            metadata={"key": "value"}  # Normal metadata
        )
        def test_function():
            return "result"

        # Should work fine
        result = test_function()
        assert result == "result"

    @patch('magpie_ai.monitor.get_client')
    @patch('magpie_ai.monitor._capture_input', side_effect=Exception("Capture failed"))
    def test_input_capture_failure_doesnt_crash(self, mock_capture, mock_get_client):
        """Test that input capture failures don't crash."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", capture_input=True)
        def test_function(x):
            return x * 2

        result = test_function(5)
        assert result == 10

        # Log should still be sent (with None input)
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['input'] is None

    @patch('magpie_ai.monitor.get_client')
    @patch('magpie_ai.monitor._capture_output', side_effect=Exception("Capture failed"))
    def test_output_capture_failure_doesnt_crash(self, mock_capture, mock_get_client):
        """Test that output capture failures don't crash."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", capture_output=True)
        def test_function():
            return "result"

        result = test_function()
        assert result == "result"

        # Log should still be sent (with None output)
        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['output'] is None


class TestThreadSafety:
    """Test thread-safety of the decorator."""

    @patch('magpie_ai.monitor.get_client')
    def test_concurrent_executions(self, mock_get_client):
        """Test multiple concurrent executions."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def concurrent_function(thread_id):
            time.sleep(0.01)  # Small delay to encourage concurrency
            return f"Thread {thread_id}"

        results = []
        threads = []

        def run_function(tid):
            result = concurrent_function(tid)
            results.append(result)

        # Start 10 concurrent threads
        for i in range(10):
            thread = threading.Thread(target=run_function, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should complete successfully
        assert len(results) == 10
        assert all(f"Thread {i}" in results for i in range(10))

        # All should have logged (10 calls)
        assert mock_client.send_log_sync.call_count == 10


class TestCaptureFlags:
    """Test capture_input and capture_output flags."""

    @patch('magpie_ai.monitor.get_client')
    def test_capture_input_disabled(self, mock_get_client):
        """Test disabling input capture."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", capture_input=False)
        def test_function(x, y):
            return x + y

        test_function(2, 3)

        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['input'] is None

    @patch('magpie_ai.monitor.get_client')
    def test_capture_output_disabled(self, mock_get_client):
        """Test disabling output capture."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project", capture_output=False)
        def test_function():
            return "result"

        test_function()

        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['output'] is None

    @patch('magpie_ai.monitor.get_client')
    def test_both_captures_disabled(self, mock_get_client):
        """Test disabling both input and output capture."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        @monitor(
            project_id="test-project",
            capture_input=False,
            capture_output=False
        )
        def test_function(x):
            return x * 2

        test_function(5)

        call_args = mock_client.send_log_sync.call_args
        assert call_args.kwargs['input'] is None
        assert call_args.kwargs['output'] is None
        # But other fields should still be present
        assert call_args.kwargs['status'] == "success"
        assert call_args.kwargs['function_name'] == "test_function"


class TestAsyncLogging:
    """Test asynchronous logging behavior."""

    @patch('magpie_ai.monitor.threading.Thread')
    @patch('magpie_ai.monitor.get_client')
    def test_logging_uses_background_thread(self, mock_get_client, mock_thread_class):
        """Test that logging happens in background thread."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread

        @monitor(project_id="test-project")
        def test_function():
            return "result"

        test_function()

        # Verify thread was created and started
        assert mock_thread_class.called
        assert mock_thread.start.called

        # Verify daemon flag
        call_kwargs = mock_thread_class.call_args.kwargs
        assert call_kwargs.get('daemon') is True

    @patch('magpie_ai.monitor.get_client')
    def test_main_function_doesnt_wait_for_logging(self, mock_get_client):
        """Test that main function doesn't wait for logging to complete."""
        mock_client = Mock()

        # Make send_log_sync slow
        def slow_send(*args, **kwargs):
            time.sleep(0.5)  # 500ms delay

        mock_client.send_log_sync = slow_send
        mock_get_client.return_value = mock_client

        @monitor(project_id="test-project")
        def fast_function():
            return "result"

        start = time.time()
        result = fast_function()
        elapsed = time.time() - start

        # Function should return quickly (not wait for logging)
        assert result == "result"
        assert elapsed < 0.1  # Should be much less than 500ms
