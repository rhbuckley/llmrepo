import logging
from typing import Any, Dict
from llmrepo.tools import BaseTool, ToolParameter, BaseToolbox


class GoogleSearchTool(BaseTool):
    """
    A tool for searching the web using Google.
    """
    name: str = "google_search"
    description: str = "Search the web using Google's search engine"
    parameters: Dict[str, ToolParameter] = {
        "query": ToolParameter(
            name="query",
            type="string",
            description="The search query to send to Google"
        )
    }

    def invoke(self, query: str) -> str:
        """
        Searches the web using Google.
        """
        return f"Searching Google for: {query}"


class BingSearchTool(BaseTool):
    """
    A tool for searching the web using Bing.
    """
    name: str = "bing_search"
    description: str = "Search the web using Bing's search engine"
    parameters: Dict[str, ToolParameter] = {
        "query": ToolParameter(
            name="query",
            type="string",
            description="The search query to send to Bing"
        )
    }

    def invoke(self, query: str) -> str:
        """
        Searches the web using Bing.
        """
        return f"Searching Bing for: {query}"

class SearchToolbox(BaseToolbox):
    """
    A toolbox for searching the web.
    """
    google_search = GoogleSearchTool()
    bing_search = BingSearchTool()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on("ainvoke:after", self._log_tool_result)
        self.on("invoke:after", self._log_tool_result)

    def _log_tool_result(self, tool: BaseTool, result: Any, *args, **kwargs) -> None:
        """
        Logs the result of a tool's execution.
        """
        logging.info(f"Tool {tool.name} returned: {result}")

