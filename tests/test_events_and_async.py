import pytest
import asyncio
from llmrepo.tools import BaseTool, ToolParameter, ToolEvent

class AsyncTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="async_tool",
            description="An async test tool",
            parameters={
                "delay": ToolParameter(
                    name="delay",
                    type="float",
                    description="Delay in seconds",
                    required=True
                )
            }
        )
    
    def invoke(self, delay: float) -> str:
        return f"Sync waited {delay}s"
    
    async def ainvoke(self, delay: float) -> str:
        await asyncio.sleep(delay)
        return f"Async waited {delay}s"

@pytest.mark.asyncio
async def test_async_execution():
    tool = AsyncTool()
    result = await tool.ainvoke(delay=0.1)
    assert result == "Async waited 0.1s"

def test_event_handlers():
    tool = AsyncTool()
    events_triggered = []
    
    def on_before(tool, **kwargs):
        events_triggered.append(("before", kwargs))
    
    def on_after(tool, result, **kwargs):
        events_triggered.append(("after", result))
    
    def on_error(tool, error, **kwargs):
        events_triggered.append(("error", str(error)))
    
    tool.on(ToolEvent.BEFORE_INVOKE, on_before)
    tool.on(ToolEvent.AFTER_INVOKE, on_after)
    tool.on(ToolEvent.ERROR, on_error)
    
    # Test successful execution
    tool.invoke(delay=0.1)
    assert len(events_triggered) == 2
    assert events_triggered[0][0] == "before"
    assert events_triggered[1][0] == "after"
    
    # Test error handling
    events_triggered.clear()
    with pytest.raises(TypeError):
        tool.invoke(delay="invalid")
    assert len(events_triggered) == 2
    assert events_triggered[0][0] == "before"
    assert events_triggered[1][0] == "error"

@pytest.mark.asyncio
async def test_async_event_handlers():
    tool = AsyncTool()
    events_triggered = []
    
    async def async_before(tool, **kwargs):
        events_triggered.append(("before", kwargs))
    
    async def async_after(tool, result, **kwargs):
        events_triggered.append(("after", result))
    
    tool.on(ToolEvent.BEFORE_AINVOKE, async_before)
    tool.on(ToolEvent.AFTER_AINVOKE, async_after)
    
    await tool.ainvoke(delay=0.1)
    assert len(events_triggered) == 2
    assert events_triggered[0][0] == "before"
    assert events_triggered[1][0] == "after" 