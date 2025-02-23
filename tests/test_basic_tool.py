import pytest
from llmrepo.tools import BaseTool, ToolParameter

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters={
                "text": ToolParameter(
                    name="text",
                    type="str",
                    description="Input text",
                    required=True
                ),
                "count": ToolParameter(
                    name="count",
                    type="int",
                    description="Number of times to repeat",
                    required=False,
                    default=1
                )
            }
        )
    
    def invoke(self, text: str, count: int = 1) -> str:
        return text * count

def test_tool_basic_execution():
    tool = SimpleTool()
    result = tool.invoke(text="hello", count=2)
    assert result == "hellohello"

def test_tool_default_parameters():
    tool = SimpleTool()
    result = tool.invoke(text="hello")
    assert result == "hello"

def test_tool_parameter_validation():
    tool = SimpleTool()
    
    # Test missing required parameter
    with pytest.raises(ValueError, match="Missing required parameter: text"):
        tool.invoke(count=2)
    
    # Test invalid parameter type
    with pytest.raises(TypeError, match="must be of type str"):
        tool.invoke(text=123)
    
    # Test unexpected parameter
    with pytest.raises(ValueError, match="Unexpected parameters"):
        tool.invoke(text="hello", invalid_param=True) 