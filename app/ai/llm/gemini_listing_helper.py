import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List

import requests  # type: ignore[import-untyped]
from PIL import Image

from app.ai.llm.gateway import get_gateway

logger = logging.getLogger(__name__)


class GeminiListingVerificationHelper:
    """
    Enhanced Gemini client specifically for listing verification with multimodal capabilities.

    Uses LLMGateway so all calls are traced through Langfuse.
    """

    def __init__(self) -> None:
        self._gateway = get_gateway()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_images_with_text(
        self, images: List[str], text_content: str, analysis_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze images along with text content for comprehensive listing verification.
        """
        try:
            image_objects = self._download_images(images)
            if not image_objects:
                raise ValueError("No valid images could be processed")

            full_prompt = (
                f"{analysis_prompt}\n\n"
                f"Text Content to analyze:\n{text_content}\n\n"
                "Please analyze both the images and text content together to provide "
                "a comprehensive assessment.\n"
                "Return your response as valid JSON only, without any additional text or formatting."
            )

            trace = self._gateway.create_trace(
                name="listing-verification",
                metadata={"image_count": len(image_objects), "mode": "images+text"},
            )

            response = await self._gateway.generate_with_images(
                prompt=full_prompt,
                images=image_objects,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 8192,
                },
                trace=trace,
                span_name="verify-images-text",
            )

            return self._handle_api_response(response.text.strip())

        except Exception as e:
            return self._handle_generation_error(str(e))

    async def analyze_text_content(
        self, text_content: str, analysis_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze text content for listing verification (no images).
        """
        try:
            full_prompt = (
                f"{analysis_prompt}\n\n"
                f"Content to analyze:\n{text_content}\n\n"
                "Return your response as valid JSON only, without any additional text or formatting."
            )

            trace = self._gateway.create_trace(
                name="listing-verification",
                metadata={"mode": "text-only"},
            )

            response = await self._gateway.generate(
                prompt=full_prompt,
                generation_config={
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 4096,
                },
                trace=trace,
                span_name="verify-text",
            )

            return self._handle_api_response(response.text.strip())

        except Exception as e:
            return self._handle_generation_error(str(e))

    def create_comprehensive_analysis_prompt(self) -> str:
        """Create a comprehensive analysis prompt for listing verification"""
        return """
You are an AI expert in rental property listing verification. Analyze the provided content (text and images if available) and evaluate it across three main categories:

1. **IMAGE/MEDIA VALIDATION** (if images provided):
   - Are images clear, well-lit, and high quality?
   - Do images show actual property spaces (not stock photos)?
   - Are images appropriate for rental listings?
   - Do images match the described property?

2. **CONTENT RELEVANCE**:
   - Is this clearly a rental property listing?
   - Does the content match rental property category?
   - Is the information coherent and professional?
   - Are there any inappropriate or suspicious elements?

3. **COMPLETENESS & QUALITY**:
   - Is essential information provided (price, location, description)?
   - Is the description detailed and informative?
   - Are important details missing?
   - Is the overall quality sufficient for a good listing?

Return a JSON response with this exact structure (keep messages concise, max 100 chars each):
{
    "image_analysis": {
        "is_valid": boolean,
        "quality_score": float (0-1),
        "issues": ["max 3 brief issues"],
        "total_images_analyzed": integer
    },
    "content_analysis": {
        "is_rental_related": boolean,
        "category_match": boolean,
        "content_score": float (0-1),
        "issues": ["max 2 brief issues"],
        "violations": [{"category": "string", "severity": "low|medium|high|critical", "message": "brief message"}]
    },
    "completeness_analysis": {
        "is_complete": boolean,
        "completeness_score": float (0-1),
        "missing_fields": ["max 3 field names"],
        "quality_issues": ["max 2 brief issues"],
        "suggestions": [{"category": "string", "message": "brief suggestion", "priority": "low|medium|high"}]
    },
    "overall_assessment": {
        "is_valid": boolean,
        "overall_score": float (0-1),
        "confidence": float (0-1),
        "major_concerns": ["max 2 primary issues"],
        "recommendations": ["max 2 brief recommendations"]
    }
}

Be thorough but CONCISE. Keep all text fields short and focused.
"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_images(image_urls: List[str], max_images: int = 8) -> List[Any]:
        """Download and preprocess images from URLs."""
        image_objects = []
        for i, image_url in enumerate(image_urls[:max_images]):
            try:
                resp = requests.get(image_url, timeout=10)
                if resp.status_code != 200:
                    logger.warning("Failed to download image %d: HTTP %s", i + 1, resp.status_code)
                    continue

                pil_image: Image.Image = Image.open(BytesIO(resp.content))
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                if pil_image.width > 2048 or pil_image.height > 2048:
                    pil_image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                image_objects.append(pil_image)
                logger.info("Successfully processed image %d", i + 1)
            except Exception as e:
                logger.error("Error processing image %d: %s", i + 1, e)

        return image_objects

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """Parse JSON response with fallback handling."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON object from response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try fixing incomplete JSON
        if response_text.count("{") > response_text.count("}"):
            missing = response_text.count("{") - response_text.count("}")
            try:
                return json.loads(response_text + "}" * missing)
            except json.JSONDecodeError:
                pass

        logger.error("All JSON parsing attempts failed")
        return {
            "error": "Failed to parse AI response",
            "raw_response": response_text[:500],
            "analysis_completed": False,
        }

    @classmethod
    def _handle_api_response(cls, response_text: str) -> Dict[str, Any]:
        """Handle and parse API response."""
        logger.info("Raw Gemini response: %s...", response_text[:500])

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        return cls._parse_json_response(response_text)

    @staticmethod
    def _handle_generation_error(error_msg: str) -> Dict[str, Any]:
        """Handle generation errors with appropriate responses."""
        logger.error("Error in analysis: %s", error_msg)

        if "quota exceeded" in error_msg.lower() or "429" in error_msg:
            return {"error": "quota exceeded", "analysis_completed": False}
        elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return {"error": "invalid api key", "analysis_completed": False}
        else:
            return {"error": error_msg, "analysis_completed": False}
