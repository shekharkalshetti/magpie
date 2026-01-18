# Magpie AI

**Enterprise-grade LLM middleware for monitoring and metadata tracking**

[![PyPI version](https://img.shields.io/pypi/v/magpie-ai.svg)](https://pypi.org/project/magpie-ai/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Magpie AI is a powerful Python SDK that seamlessly integrates with your LLM applications to provide comprehensive monitoring, cost tracking, PII detection, and content moderation capabilities.

## Features

✨ **Zero-Code Integration** - Add monitoring with a single decorator  
💰 **Cost Tracking** - Automatic token counting and cost calculation  
🔐 **PII Detection** - Detect and redact sensitive data automatically  
🛡️ **Content Moderation** - Policy-based content validation  
📊 **Comprehensive Metrics** - Track latency, tokens, costs, and custom data  
⚡ **Non-Blocking** - Asynchronous logging that never crashes your app  
🔄 **Framework Agnostic** - Works with OpenAI, Anthropic, and any LLM

## Installation

```bash
pip install magpie-ai
```

## Quick Start

### Basic Usage

Wrap any LLM function with `@magpie_ai.monitor`:

```python
import magpie_ai
from openai import OpenAI

@magpie_ai.monitor(
    project_id="my-project",
    model="gpt-4"
)
def chat_with_gpt(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Use it exactly as before - monitoring is automatic!
result = chat_with_gpt("What is Python?")
```

### Advanced Configuration

```python
@magpie_ai.monitor(
    project_id="my-project",
    model="gpt-4",                          # Auto-lookup pricing
    pii=True,                               # Enable PII detection
    content_moderation=True,                # Enable content moderation
    custom={                                # Add custom metadata
        "user_id": "user_123",
        "session": "chat_abc",
        "department": "sales"
    }
)
def llm_function(prompt: str) -> str:
    # Your LLM code here
    pass
```

## Core Features

### 1. Cost Tracking

Automatic token counting and USD cost calculation:

```python
# Costs are calculated automatically
@magpie_ai.monitor(
    project_id="my-project",
    model="gpt-4-turbo"  # Pricing looked up automatically
)
def expensive_llm_call(text: str) -> str:
    # Tracked metrics include:
    # - input_tokens, output_tokens, total_tokens
    # - input_cost, output_cost (in USD)
    pass
```

**Supported Models:**

- OpenAI: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `gpt-4o`, etc.
- Anthropic: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`, etc.
- Custom pricing with `input_token_price` and `output_token_price`

### 2. PII Detection & Redaction

Automatically detect and redact sensitive information:

```python
@magpie_ai.monitor(
    project_id="my-project",
    pii=True  # Enable PII detection
)
def process_user_input(data: str) -> str:
    # Input is scanned for:
    # - Email addresses
    # - Phone numbers
    # - Credit cards
    # - SSNs
    # - Names and addresses
    # Automatically redacted before sending to LLM
    pass
```

### 3. Content Moderation

Policy-based content validation:

```python
from magpie_ai import ContentModerationError

@magpie_ai.monitor(
    project_id="my-project",
    content_moderation=True  # Enable policy checking
)
def moderated_llm_call(prompt: str) -> str:
    pass

try:
    result = moderated_llm_call("Your prompt here")
except ContentModerationError as e:
    print(f"Content blocked: {e}")
    # Handle moderation failure gracefully
```

### 4. Custom Metadata

Attach arbitrary metadata to each execution:

```python
@magpie_ai.monitor(
    project_id="my-project",
    custom={
        "user_id": "john_doe",
        "department": "engineering",
        "client": "acme_corp",
        "version": "2.1.0"
    }
)
def track_with_context(prompt: str) -> str:
    pass
```

### 5. Comprehensive Metrics

Every execution captures:

```
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "project_id": "my-project",
  "timestamp": "2025-01-18T21:30:00Z",
  "total_latency_ms": 1240,
  "input_tokens": 156,
  "output_tokens": 243,
  "total_tokens": 399,
  "input_cost": 0.00468,
  "output_cost": 0.00729,
  "context_utilization_percent": 45.2,
  "custom": {...}
}
```

## Pricing Configuration

### Option 1: Model-based (Recommended)

```python
@magpie_ai.monitor(
    project_id="my-project",
    model="gpt-4"  # Pricing auto-looked up
)
def my_function():
    pass
```

### Option 2: Custom Pricing

```python
@magpie_ai.monitor(
    project_id="my-project",
    input_token_price=0.03,      # $0.03 per 1M input tokens
    output_token_price=0.06      # $0.06 per 1M output tokens
)
def my_function():
    pass
```

**Note:** Cannot use both `model` and explicit pricing parameters together.

## Error Handling

Magpie AI **fails open** - it never crashes your application:

```python
@magpie_ai.monitor(project_id="my-project")
def critical_function():
    # Even if monitoring fails, your function runs normally
    # Errors are logged but never raised
    pass
```

## Validation

The monitor decorator validates all parameters:

```python
# ✓ Valid
@magpie_ai.monitor(project_id="my-project", model="gpt-4")

# ✗ Error: project_id is required
@magpie_ai.monitor()

# ✗ Error: cannot use both model and explicit pricing
@magpie_ai.monitor(
    project_id="my-project",
    model="gpt-4",
    input_token_price=0.03
)

# ✗ Error: custom must be a dict
@magpie_ai.monitor(project_id="my-project", custom="invalid")
```

## Thread Safety

Magpie AI is fully thread-safe and can be used in multi-threaded applications:

```python
import threading

@magpie_ai.monitor(project_id="my-project")
def concurrent_function():
    pass

threads = [
    threading.Thread(target=concurrent_function)
    for _ in range(10)
]
for thread in threads:
    thread.start()
    thread.join()
```

## Context Manager

Use Magpie AI as a context manager for fine-grained control:

```python
from magpie_ai import context

def my_function():
    with context(project_id="my-project", custom={"session": "abc"}):
        # Code here is monitored
        result = call_llm()

    # Monitoring stops when context exits
    return result
```

## Supported LLM Providers

- ✅ **OpenAI** - GPT-3.5, GPT-4, GPT-4 Turbo, GPT-4o
- ✅ **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- ✅ **Custom/Local** - Any provider with token extraction
- ✅ **Framework Agnostic** - Works with LangChain, LlamaIndex, and others

## API Reference

### `@magpie_ai.monitor()`

Main decorator for LLM monitoring.

**Parameters:**

- `project_id` (str, required): Project identifier
- `model` (str, optional): Model name for auto pricing lookup
- `input_token_price` (float, optional): Price per 1M input tokens
- `output_token_price` (float, optional): Price per 1M output tokens
- `custom` (dict, optional): Custom metadata (must be JSON-serializable)
- `pii` (bool, default=False): Enable PII detection and redaction
- `content_moderation` (bool, default=False): Enable content moderation
- `capture_input` (bool, default=True): Capture function inputs
- `trace_id` (str, optional): Custom trace ID (auto-generated if not provided)
- `llm_url` (str, default="http://localhost:1234"): LM Studio URL for PII/moderation
- `llm_model` (str, default="qwen2.5-1.5b-instruct"): Model for PII/moderation analysis

**Raises:**

- `ValueError`: If project_id is empty or pricing config is invalid
- `TypeError`: If custom is not a dict or pricing values aren't numeric

### `magpie_ai.context()`

Context manager for monitoring specific code blocks.

**Parameters:** Same as `@monitor()` decorator

**Returns:** Context manager that enables monitoring within its scope

### `magpie_ai.ContentModerationError`

Exception raised when content moderation blocks a request.

```python
from magpie_ai import ContentModerationError

try:
    result = monitored_function()
except ContentModerationError as e:
    print(f"Blocked: {e}")
```

## Examples

### Example 1: E-Commerce Product Search

```python
import magpie_ai
from openai import OpenAI

@magpie_ai.monitor(
    project_id="ecommerce-ai",
    model="gpt-3.5-turbo",
    custom={"service": "product-search"}
)
def search_products(query: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# Usage
result = search_products("Find me blue running shoes under $100")
print(result)
```

### Example 2: Sensitive Data Processing with PII Detection

```python
@magpie_ai.monitor(
    project_id="secure-ai",
    model="gpt-4",
    pii=True,  # Detect and redact PII
    custom={"compliance": "HIPAA"}
)
def analyze_patient_notes(notes: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": notes}]
    )
    return response.choices[0].message.content

# Sensitive data like SSNs, emails are automatically redacted
result = analyze_patient_notes("Patient John Doe (SSN: 123-45-6789)...")
```

### Example 3: Multi-Step Workflow

```python
@magpie_ai.monitor(
    project_id="workflow",
    model="gpt-4-turbo",
    pii=True,
    content_moderation=True
)
def generate_report(data: str) -> str:
    # Step 1: Analyze
    analysis = analyze_data(data)

    # Step 2: Generate
    report = generate_markdown(analysis)

    # Step 3: Return
    return report
```

## Performance

- **Latency:** < 5ms overhead per call (non-blocking)
- **Memory:** ~2MB per monitor instance
- **Throughput:** Supports 1000+ concurrent monitored calls

## Troubleshooting

### Import Error: No module named 'magpie_ai'

```bash
pip install --upgrade magpie-ai
```

### PII Detection Not Working

Ensure LM Studio is running:

```bash
llm_url="http://localhost:1234"
llm_model="qwen2.5-1.5b-instruct"
```

### ContentModerationError

Check your moderation policy configuration in the backend.

### High Latency

PII detection and content moderation add ~200-500ms. Consider enabling only when needed.

## License

MIT - See [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md)

## Support

- 📧 **Email:** team@magpie.dev
- 📚 **Docs:** https://docs.magpie.dev
- 🐛 **Issues:** https://github.com/magpie-ai/sdk/issues

## Changelog

### v0.2.1 (Jan 18, 2025)

- ✅ Fixed module import path (magpie_ai)
- ✅ Added proper type hints for decorators
- ✅ Added parameter validation
- ✅ Improved error messages
- ✅ Updated documentation

### v0.2.0 (Jan 18, 2025)

- ✅ Renamed package to magpie-ai
- ✅ Full type support with py.typed
- ✅ Generic decorator types

### v0.1.0 (Dec 26, 2024)

- ✅ Initial release
- ✅ Core monitoring capabilities
- ✅ PII detection and redaction
- ✅ Content moderation
- ✅ Cost tracking

---

Built with ❤️ by [Magpie Team](https://magpie.dev)
