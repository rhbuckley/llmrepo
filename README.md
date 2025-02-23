# 🌟 LLMRepo - Structured Tool Integration for Large Language Models

[![PyPI Version](https://img.shields.io/pypi/v/llmtools)](https://pypi.org/project/llmtools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**LLMRepo.tools** is your swiss army knife 🛠️ for building structured, type-safe tool integrations with Large Language Models (LLMs). Designed specifically for modern AI workflows, it provides seamless integration with platforms like OpenAI while offering enterprise-grade validation and monitoring capabilities.

## 🚀 Features

-   **Type-Safe Tools** 🔒 - Pydantic-powered parameter validation
-   **Observability** 📊 - Built-in event system for monitoring
-   **Async First** ⚡ - Native support for asynchronous operations
-   **Context Management** 🧠 - Shared state across tool executions
-   **OpenAI Ready** 🤖 - Automatic format conversion for function calling
-   **Modular Design** 🧱 - Toolbox system for organizing tool collections

## 📦 Installation

```bash
pip install llmrepo
```

## 🎯 Quick Start

### Create Your First Tool

```python
from llmrepo.tools import BaseTool, ToolParameter

class WeatherTool(BaseTool):
    """Get current weather conditions"""

    name = "get_weather"
    description = "Fetch current weather data for any location"

    parameters = {
        "location": ToolParameter(
            name="location",
            type="string",
            description="City and country (e.g., 'London, UK')",
            required=True
        ),
        "units": ToolParameter(
            name="units",
            type="string",
            description="Temperature units system",
            enum=["celsius", "fahrenheit"],
            default="celsius"
        )
    }

    def invoke(self, location: str, units: str = "celsius") -> str:
        """Actual implementation would call weather API here"""
        return f"Weather in {location}: 22°{units[0].upper()}"
```

### Execute and Monitor

```python
tool = WeatherTool()

# Attach event listeners
tool.on("invoke:before", lambda: print("🌤️ Checking weather..."))
tool.on("invoke:after", lambda result: print(f"Result: {result}"))

# Call your tool
print(tool.invoke("Paris, France"))  # Output: Weather in Paris: 22°C
```

## 🧠 Core Concepts

### 🔧 Tools Architecture

Define atomic operations with strict input validation:

```python
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical operations"

    parameters = {
        "numbers": ToolParameter(
            type="array",
            description="List of operands",
            required=True
        ),
        "operation": ToolParameter(
            type="string",
            enum=["add", "subtract", "multiply"],
            required=True
        )
    }

    def invoke(self, numbers: list[float], operation: str) -> float:
        match operation:
            case 'add': return sum(numbers)
            case 'multiply': return math.prod(numbers)
            case _: raise ValueError("Invalid operation")
```

### 🧰 Toolbox Orchestration

Group related tools and share context:

```python
from llmrepo.tools import BaseToolbox

class PhysicsToolbox(BaseToolbox):
    weather = WeatherTool()
    calc = CalculatorTool()

    def init(self, api_key: str):
        super().__init__()
        self.context["api_key"] = api_key  # Shared across tools

# Usage
toolbox = PhysicsToolbox(api_key="my-secret-key")
toolbox.weather.invoke("Berlin, DE")
```

Tools can still have their own context, and this context will be merged with the toolbox's context.

```python
# only updates the weather tool's context
toolbox.weather.context["api_key"] = "my-secret-key"

# only updates the calc tool's context
toolbox.calc.context["api_key"] = "my-secret-key"

# updates both contexts
toolbox.context["api_key"] = "my-secret-key"
toolbox.context["api_key"] = "my-secret-key"
```

If toolbox is initialized with a context, this context can be changed by setting the `context` on the toolbox or the tool (since it was already defined).

```python
toolbox = PhysicsToolbox(context={"api_key": "my-secret-key"})

# this updates the context of api_key for all tools
toolbox.weather.context["api_key"] = "my-secret-key"
```

### 🔔 Event System

Monitor tool lifecycle events:

```python
def log_usage(tool_name: str):
    print(f"📡 {tool_name} triggered!")

toolbox.on("invoke:before", lambda tool: log_usage(tool.name))
```

### Available Events

| Event              | Description                          |
| ------------------ | ------------------------------------ |
| `invoke:before`    | Pre-sync execution hook              |
| `invoke:after`     | Post-sync execution hook             |
| `ainvoke:before`   | Pre-async execution hook             |
| `ainvoke:after`    | Post-async execution hook            |
| `validation_error` | Parameter validation failure         |
| `runtime_error`    | Unhandled exception during execution |

## ⚡ Advanced Patterns

### Async Superpowers

```python
class AsyncSearchTool(BaseTool):
    async def ainvoke(self, query: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://api.com/search?q={query}") as response:
                return await response.json()
```

### OpenAI Integration

```python
# Convert toolbox to OpenAI-compatible format
functions = PhysicsToolbox().as_openai_tools()

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's 18°C in Fahrenheit?"}],
    functions=functions,
    function_call={"name": "temperature_converter"}
)
```

### Context-Aware Execution

```python
class PersonalizedGreeter(BaseTool):
    def invoke(self) -> str:
        user = self.context.get("user")
        return f"Hello {user['name']}! You have {user['messages']} unread messages."

toolbox = PhysicsToolbox(context={"user": {"name": "Alice", "messages": 3}})
print(toolbox.personalized_greeter.invoke())  # Hello Alice! You have 3 unread messages.
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository 🍴
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 🌟

Please ensure tests pass using `pytest tests/` and update documentation accordingly.

### Running Tests

```bash
# if you are developing locally
pip install -e .

# run tests
pytest tests/
```

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

Built with ❤️ by AI enthusiasts | Documentation improvements welcome!
