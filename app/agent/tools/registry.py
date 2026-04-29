import logging
from typing import Any, Dict, List, Optional

from app.agent.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all agent tools.

    Responsibilities:
    - Hold tool instances keyed by name
    - Build the Gemini `Tool` object (function declarations) for the LLM
    - Dispatch tool execution by name and return structured results
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool. Overwrites silently if the name already exists."""
        if tool.name in self._tools:
            logger.warning("Tool '%s' is already registered — overwriting.", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get_tool(self) -> Any:
        """
        Build and return a google-genai `Tool` object containing all registered
        FunctionDeclarations. Pass this in `GenerateContentConfig(tools=[...])`.
        """
        from google.genai import types  # type: ignore[import]

        declarations = [t.to_function_declaration() for t in self._tools.values()]
        return types.Tool(function_declarations=declarations)

    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatch a tool call by name.

        Args:
            name: Tool name as returned by the LLM function call.
            args: Keyword arguments extracted from the function call.
            context: Optional execution context (e.g. user_id, auth_token)
                     passed through to the tool's execute() as a "context" kwarg.

        Returns:
            Tool result dict. Always contains "status": "success" | "error".
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.error("LLM called unknown tool: '%s'", name)
            return {"status": "error", "error": f"Unknown tool: {name}"}

        logger.info("Executing tool '%s' with args: %s", name, args)
        try:
            kwargs = {**args}
            if context is not None:
                kwargs["context"] = context
            result = await tool.execute(**kwargs)
            logger.info("Tool '%s' completed successfully.", name)
            return result
        except Exception as e:
            logger.error("Tool '%s' raised an exception: %s", name, e, exc_info=True)
            return {"status": "error", "error": str(e)}

    def list_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)
