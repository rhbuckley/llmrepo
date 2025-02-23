import pytest
from llmrepo.tools import BaseTool, ToolParameter

pytest_plugins = ("pytest_asyncio",)

@pytest.fixture
def simple_tool():
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
                    )
                }
            )
        
        def invoke(self, text: str) -> str:
            return text.upper()
    
    return SimpleTool() 