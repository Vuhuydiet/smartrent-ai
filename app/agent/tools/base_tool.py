from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Each concrete tool must declare its name and description as class attributes,
    implement `to_function_declaration()` to expose its schema to the LLM, and
    implement `execute()` to perform the actual work.
    """

    name: str
    description: str

    @abstractmethod
    def to_function_declaration(self) -> Any:
        """
        Return a Vertex AI FunctionDeclaration describing this tool.

        The declaration is used by the LLM to decide when and how to call the tool.
        """
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the tool with the arguments the LLM provided.

        Args:
            **kwargs: Parameters matching the schema in `to_function_declaration`.

        Returns:
            A dict that is sent back to the LLM as the FunctionResponse payload.
            Always include a "status" key ("success" | "error") so the orchestrator
            can detect failures without inspecting every tool's response shape.
        """
        ...
