from typing import Optional
from app.ai.llm.gemini_client import GeminiClient
from app.ai.llm.base_llm import BaseLLM
from app.core.config import settings


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_llm(llm_type: str = "gemini", model_name: Optional[str] = None) -> BaseLLM:
        """Create an LLM instance based on the specified type."""
        if llm_type.lower() == "gemini":
            model_name = model_name or "gemini-2.5-pro"
            return GeminiClient(model_name)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


# Global LLM instance - initialized once and reused
_llm_instance: Optional[BaseLLM] = None


def get_llm_instance() -> BaseLLM:
    """Get the global LLM instance, creating it if it doesn't exist."""
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = LLMFactory.create_llm("gemini")
    
    return _llm_instance


def reset_llm_instance() -> None:
    """Reset the global LLM instance (useful for testing)."""
    global _llm_instance
    _llm_instance = None
