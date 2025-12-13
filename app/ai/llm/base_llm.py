from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def generate_response(self, conversation_context: str) -> str:
        """Generate response from the LLM."""
        pass

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name
