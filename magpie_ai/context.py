"""
Context manager for monitoring code blocks.

Provides thread-local metadata storage that can be used standalone
or in combination with the @monitor() decorator.

Usage:
    # Standalone - logs the context execution
    with magpie_ai.context(project_id="my-project", metadata={"user_id": "123"}):
        result = llm.call(prompt)
    
    # Combined with decorator - metadata is merged
    with magpie_ai.context(metadata={"user_id": "123"}):
        @magpie_ai.monitor(project_id="my-project", metadata={"model": "gpt-4"})
        def my_function():
            pass
        my_function()  # Logs with both context and decorator metadata
"""
import threading
from contextlib import contextmanager
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from magpie_ai.client import get_client
from magpie_ai.validation import sanitize_metadata


# Thread-local storage for context metadata
_context_storage = threading.local()


class ContextMetadata:
    """Container for thread-local context metadata."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        self.project_id = project_id
        self.metadata = metadata or {}
        self.trace_id = trace_id


def get_context_metadata() -> Optional[ContextMetadata]:
    """
    Get the current thread-local context metadata.

    Returns:
        ContextMetadata if a context is active, None otherwise
    """
    return getattr(_context_storage, 'metadata', None)


def set_context_metadata(context_metadata: Optional[ContextMetadata]):
    """
    Set the current thread-local context metadata.

    Args:
        context_metadata: The context metadata to set, or None to clear
    """
    _context_storage.metadata = context_metadata


def merge_metadata(
    decorator_metadata: Optional[Dict[str, Any]],
    context_metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge decorator and context metadata.

    Context metadata takes precedence over decorator metadata.

    Args:
        decorator_metadata: Metadata from @monitor() decorator
        context_metadata: Metadata from context manager

    Returns:
        Merged metadata dictionary
    """
    merged = {}

    if decorator_metadata:
        merged.update(decorator_metadata)

    if context_metadata:
        merged.update(context_metadata)

    return merged


@contextmanager
def context(
    project_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    log_context: bool = True
):
    """
    Context manager for monitoring code execution.

    Stores metadata in thread-local storage that can be accessed by
    @monitor() decorators. Optionally logs the context execution itself.

    Args:
        project_id: Optional project ID (required if log_context=True)
        metadata: Metadata to attach to executions within this context
        trace_id: Optional trace ID (generated if not provided and log_context=True)
        log_context: Whether to log the context execution itself (default: True)

    Example (standalone):
        with triton.context(
            project_id="my-project",
            metadata={"user_id": "123", "session": "abc"}
        ):
            result = llm.call(prompt)

    Example (with decorator):
        with magpie_ai.context(metadata={"user_id": "123"})::
            @triton.monitor(project_id="my-project", metadata={"model": "gpt-4"})
            def my_function():
                pass

            my_function()  # Logs with merged metadata

    Note:
        - Thread-safe: Each thread has its own context
        - Nestable: Inner contexts override outer contexts
        - Fail-open: Never crashes your application
        - Context metadata takes precedence over decorator metadata
    """
    # Get previous context (for nesting)
    previous_context = get_context_metadata()

    # Sanitize metadata
    sanitized_metadata = None
    if metadata:
        try:
            sanitized_metadata = sanitize_metadata(metadata)
        except Exception:
            # Fail open - continue without metadata
            sanitized_metadata = None

    # Create new context metadata
    new_context = ContextMetadata(
        project_id=project_id,
        metadata=sanitized_metadata,
        trace_id=trace_id
    )

    # Set in thread-local storage
    set_context_metadata(new_context)

    # If logging the context execution
    started_at = None
    error_message = None
    status = "success"
    execution_trace_id = None

    if log_context:
        started_at = datetime.utcnow()
        execution_trace_id = trace_id or str(uuid.uuid4())

    try:
        yield

    except Exception as e:
        if log_context:
            status = "error"
            error_message = str(e)
        raise  # Re-raise exception

    finally:
        # Restore previous context
        set_context_metadata(previous_context)

        # Log the context execution if requested
        if log_context and project_id and started_at:
            completed_at = datetime.utcnow()
            duration_ms = int(
                (completed_at - started_at).total_seconds() * 1000)

            # Send log in background thread
            try:
                import threading as th

                def send_context_log():
                    try:
                        client = get_client()
                        client.send_log_sync(
                            project_id=project_id,
                            metadata=sanitized_metadata,
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            status=status,
                            error_message=error_message,
                            trace_id=execution_trace_id,
                            function_name="<context>"
                        )
                    except Exception:
                        # Fail open - silently ignore errors
                        pass

                thread = th.Thread(target=send_context_log, daemon=True)
                thread.start()

            except Exception:
                # Fail open - don't crash user's code
                pass
