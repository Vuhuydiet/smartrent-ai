import json
import logging
from io import BytesIO
from typing import Any, Dict, List

import google.generativeai as genai
import requests  # type: ignore[import-untyped]
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiListingVerificationHelper:
    """
    Enhanced Gemini client specifically for listing verification with multimodal capabilities
    """

    def __init__(self) -> None:
        """Initialize the Gemini listing verification helper"""
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured")

        logger.info(
            f"Configuring Gemini API with key: {settings.GEMINI_API_KEY[:10]}..."
        )
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Use latest models available in the new account
        self.vision_model = genai.GenerativeModel("gemini-2.5-flash")
        self.text_model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Gemini models initialized successfully")

    def _process_image(self, image_url: str, index: int) -> Image.Image:
        """Process a single image from URL"""
        response = requests.get(image_url, timeout=10)
        if response.status_code != 200:
            raise ValueError(
                f"Failed to download image {index+1}: HTTP {response.status_code}"
            )

        pil_image: Image.Image = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Resize if too large (max 2048x2048)
        if pil_image.width > 2048 or pil_image.height > 2048:
            pil_image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

        return pil_image

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response with fallback handling"""
        try:
            result = json.loads(response_text)
            logger.info("Successfully parsed JSON response from Gemini")
            return result
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse Gemini response as JSON: {response_text[:200]}..."
            )
            logger.error(f"JSON error: {str(e)}")

            # Try to fix common JSON issues
            import re

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    logger.info("Successfully extracted JSON from response")
                    return result
                except json.JSONDecodeError:
                    pass

            # Try to fix incomplete JSON by adding closing braces
            if response_text.count("{") > response_text.count("}"):
                missing_braces = response_text.count("{") - response_text.count("}")
                fixed_text = response_text + "}" * missing_braces
                try:
                    result = json.loads(fixed_text)
                    logger.info("Successfully fixed incomplete JSON")
                    return result
                except json.JSONDecodeError:
                    pass

            logger.error("All JSON parsing attempts failed")
            return {
                "error": "Failed to parse AI response",
                "raw_response": response_text[:500],
                "analysis_completed": False,
            }

    def _handle_api_response(self, response_text: str) -> Dict[str, Any]:
        """Handle and parse API response"""
        logger.info(f"Raw Gemini response: {response_text[:500]}...")

        # Clean up response if it contains markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        return self._parse_json_response(response_text)

    def _generate_content_with_error_handling(
        self, content_parts: List[Any], generation_config: Any
    ) -> Dict[str, Any]:
        """Generate content with comprehensive error handling"""
        try:
            response = self.vision_model.generate_content(
                content_parts, generation_config=generation_config
            )
            return self._handle_api_response(response.text.strip())
        except Exception as e:
            return self._handle_generation_error(str(e))

    def _handle_generation_error(self, error_msg: str) -> Dict[str, Any]:
        """Handle generation errors with appropriate responses"""
        logger.error(f"Error in multimodal analysis: {error_msg}")

        # Check if it's a quota or API key issue
        if "quota exceeded" in error_msg.lower() or "429" in error_msg:
            return {"error": "quota exceeded", "analysis_completed": False}
        elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return {"error": "invalid api key", "analysis_completed": False}
        else:
            return {"error": error_msg, "analysis_completed": False}

    async def analyze_images_with_text(  # noqa: C901
        self, images: List[str], text_content: str, analysis_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze images along with text content for comprehensive listing verification

        Args:
            images: List of image URLs
            text_content: Text content to analyze alongside images
            analysis_prompt: Specific prompt for analysis

        Returns:
            Dict containing analysis results
        """
        try:
            # Prepare images
            image_objects = []
            for i, image_url in enumerate(
                images[:8]
            ):  # Limit to 8 images for API constraints
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        pil_image: Image.Image = Image.open(BytesIO(response.content))
                        # Convert to RGB if necessary
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # Resize if too large (max 2048x2048)
                        if pil_image.width > 2048 or pil_image.height > 2048:
                            pil_image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                        image_objects.append(pil_image)
                        logger.info(f"Successfully processed image {i+1}")
                    else:
                        logger.warning(
                            f"Failed to download image {i+1}: HTTP {response.status_code}"
                        )
                except Exception as e:
                    logger.error(f"Error processing image {i+1}: {str(e)}")

            if not image_objects:
                raise ValueError("No valid images could be processed")

            # Create comprehensive prompt
            full_prompt = f"""
{analysis_prompt}

Text Content to analyze:
{text_content}

Please analyze both the images and text content together to provide a comprehensive assessment.
Return your response as valid JSON only, without any additional text or formatting.
"""

            # Generate content with both images and text
            content_parts = [full_prompt] + image_objects

            response = self.vision_model.generate_content(
                content_parts,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent results
                    top_p=0.8,
                    top_k=20,
                    max_output_tokens=8192,  # Increased significantly for complete response
                ),
            )

            # Parse response
            response_text = response.text.strip()
            logger.info(f"Raw Gemini response: {response_text[:500]}...")

            # Clean up response if it contains markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            try:
                result = json.loads(response_text)
                logger.info("Successfully parsed JSON response from Gemini")
                return result  # type: ignore[no-any-return]
            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse Gemini response as JSON: {response_text[:200]}..."
                )
                logger.error(f"JSON error: {str(e)}")

                # Try to fix common JSON issues
                fixed_text = response_text

                # Try to find and extract complete JSON
                import re

                json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        logger.info("Successfully extracted JSON from response")
                        return result
                    except json.JSONDecodeError:
                        pass

                # Try to fix incomplete JSON by adding closing braces
                if response_text.count("{") > response_text.count("}"):
                    missing_braces = response_text.count("{") - response_text.count("}")
                    fixed_text = response_text + "}" * missing_braces
                    try:
                        result = json.loads(fixed_text)
                        logger.info("Successfully fixed incomplete JSON")
                        return result
                    except json.JSONDecodeError:
                        pass

                logger.error("All JSON parsing attempts failed")
                # Return a fallback response
                return {
                    "error": "Failed to parse AI response",
                    "raw_response": response_text[:500],
                    "analysis_completed": False,
                }

        except Exception as e:
            logger.error(f"Error in multimodal analysis: {str(e)}")
            error_msg = str(e)

            # Check if it's a quota or API key issue
            if "quota exceeded" in error_msg.lower() or "429" in error_msg:
                return {"error": "quota exceeded", "analysis_completed": False}
            elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return {"error": "invalid api key", "analysis_completed": False}
            else:
                return {"error": error_msg, "analysis_completed": False}

    async def analyze_text_content(
        self, text_content: str, analysis_prompt: str
    ) -> Dict[str, Any]:
        """
        Analyze text content for listing verification

        Args:
            text_content: Text content to analyze
            analysis_prompt: Specific prompt for analysis

        Returns:
            Dict containing analysis results
        """
        try:
            full_prompt = f"""
{analysis_prompt}

Content to analyze:
{text_content}

Return your response as valid JSON only, without any additional text or formatting.
"""

            response = self.text_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=20,
                    max_output_tokens=4096,  # Increased for complete response
                ),
            )

            response_text = response.text.strip()

            # Clean up response if it contains markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to parse text analysis response as JSON: {response_text}"
                )
                return {
                    "error": "Failed to parse AI response",
                    "raw_response": response_text,
                    "analysis_completed": False,
                }

        except Exception as e:
            logger.error(f"Error in text analysis: {str(e)}")
            error_msg = str(e)

            # Check if it's a quota or API key issue
            if "quota exceeded" in error_msg.lower() or "429" in error_msg:
                return {"error": "quota exceeded", "analysis_completed": False}
            elif "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
                return {"error": "invalid api key", "analysis_completed": False}
            else:
                return {"error": error_msg, "analysis_completed": False}

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
