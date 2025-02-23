import pytest
from llmrepo.tools import BaseTool, BaseToolbox, ToolParameter

class GreetingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="greet",
            description="Greets a user",
            parameters={
                "name": ToolParameter(
                    name="name",
                    type="str",
                    description="Name to greet",
                    required=True
                )
            }
        )
    
    def invoke(self, name: str) -> str:
        language = self.context.get("language", "en")
        return "Hola " + name + "!" if language == "es" else "Hello " + name + "!"

class CounterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="counter",
            description="A simple counter",
            parameters={}
        )
    
    def invoke(self) -> int:
        count = self.context.get("count", 0) + 1
        self.context["count"] = count
        return count

@pytest.fixture
def test_toolbox():
    class TestToolbox(BaseToolbox):
        greet = GreetingTool()
        counter = CounterTool()
    
    return TestToolbox(context={"language": "es"})

def test_toolbox_context_sharing(test_toolbox):
    # Test shared context
    result = test_toolbox.greet.invoke(name="Alice")
    assert result == "Hola Alice!"

    # Test individual tool context
    assert test_toolbox.counter.invoke() == 1
    assert test_toolbox.counter.invoke() == 2

    # Test context isolation
    assert "count" not in test_toolbox.greet.context

def test_toolbox_tool_discovery(test_toolbox):
    tools = test_toolbox.get_tools()
    assert len(tools) == 2
    assert any(t.name == "greet" for t in tools)
    assert any(t.name == "counter" for t in tools)

def test_openai_format(test_toolbox):
    tools = test_toolbox.as_openai_tools()
    
    assert len(tools) == 2
    assert all(tool["type"] == "function" for tool in tools)
    assert any(tool["function"]["name"] == "greet" for tool in tools)
    assert any(tool["function"]["name"] == "counter" for tool in tools) 