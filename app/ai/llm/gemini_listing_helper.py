import asyncio
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List

import httpx
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

    async def analyze_multimodal(
        self,
        image_urls: List[str],
        video_urls: List[str],
        text_content: str,
        analysis_prompt: str,
    ) -> Dict[str, Any]:
        """
        Analyze images and videos along with text content.
        """
        try:
            # Download images and videos in parallel
            images_task = self._download_images(image_urls)
            videos_task = self._download_videos(video_urls)
            image_objects, video_objects = await asyncio.gather(
                images_task, videos_task
            )

            logger.info(
                "Media download complete: %d images, %d videos loaded successfully.",
                len(image_objects),
                len(video_objects),
            )

            # Extract keyframes from videos if possible to speed up
            processed_images = list(image_objects)
            final_videos = []

            logger.info(
                "Starting media processing: %d images, %d video objects",
                len(processed_images),
                len(video_objects),
            )

            try:
                from app.utils.video_utils import extract_keyframes

                for v_data in video_objects:
                    try:
                        logger.info("Attempting keyframe extraction for a video...")
                        frames = extract_keyframes(v_data["data"])
                        if frames:
                            logger.info(
                                "Extracted %d frames from video to speed up analysis",
                                len(frames),
                            )
                            processed_images.extend(frames)
                        else:
                            logger.warning(
                                "No frames extracted from video, falling back to full video"
                            )
                            final_videos.append(v_data)
                    except Exception as ve:
                        logger.error(
                            "Keyframe extraction failed for a video: %s. Falling back.",
                            ve,
                        )
                        final_videos.append(v_data)
            except ImportError:
                logger.warning("OpenCV not found, sending full videos to Gemini")
                final_videos = video_objects
            except Exception as e:
                logger.error("Unexpected error in video processing block: %s", e)
                final_videos = video_objects

            if not processed_images and not final_videos:
                logger.warning(
                    "No media downloaded (images: %d, videos: %d). Falling back to text-only.",
                    len(image_objects),
                    len(video_objects),
                )
                return await self.analyze_text_content(text_content, analysis_prompt)

            logger.info(
                "Final payload: %d images, %d videos",
                len(processed_images),
                len(final_videos),
            )

            full_prompt = (
                "CRITICAL: YOU MUST ANALYZE ALL ATTACHED MEDIA (IMAGES/VIDEOS) CAREFULLY.\n"
                "I am providing you with actual binary media data alongside this text.\n"
                f"{analysis_prompt}\n\n"
                "### TEXT DATA TO VERIFY:\n"
                f"{text_content}\n\n"
                "### YOUR TASK:\n"
                "1. Cross-reference the provided text with the visual details in the images and videos.\n"
                "2. Check for stock photos/videos, watermarks, and consistency.\n"
                "3. Ensure the media matches the described property.\n"
                "4. Return valid JSON only."
            )

            trace = self._gateway.create_trace(
                name="listing-verification",
                metadata={
                    "image_count": len(image_objects),
                    "video_count": len(video_objects),
                    "mode": "multimodal",
                },
            )

            response = await self._gateway.generate_multimodal(
                prompt=full_prompt,
                images=processed_images,
                videos=final_videos,
                system_instruction=self.create_system_instruction(),
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
                trace=trace,
            )

            return self._handle_api_response(self._response_text(response))

        except Exception as e:
            logger.error("Multimodal analysis failed: %s", e, exc_info=True)
            return self._handle_generation_error(str(e))

    async def _download_videos(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Download videos in parallel and prepare for Gemini."""
        if not urls:
            return []

        async def _download_one(i, url, client):
            try:
                logger.info("Attempting to download video %d from: %s", i + 1, url)
                # Limit to 10MB for speed
                resp = await client.get(
                    url, headers={"Range": "bytes=0-10485760"}, timeout=20.0
                )

                if resp.status_code not in [200, 206]:
                    resp = await client.get(url, timeout=30.0)

                if resp.status_code in [200, 206] and len(resp.content) > 0:
                    mime_type = resp.headers.get("Content-Type", "video/mp4")
                    return {"data": resp.content, "mime_type": mime_type}
            except Exception as e:
                logger.error("Exception downloading video %d: %s", i + 1, str(e))
            return None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }

        async with httpx.AsyncClient(headers=headers) as client:
            tasks = [_download_one(i, url, client) for i, url in enumerate(urls)]
            results = await asyncio.gather(*tasks)

        return [v for v in results if v is not None]

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
                system_instruction=self.create_system_instruction(),
                generation_config={
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",
                },
                trace=trace,
                span_name="verify-text",
            )

            return self._handle_api_response(self._response_text(response))

        except Exception as e:
            import traceback

            logger.error(
                "[analyze_text_content] FULL ERROR:\n%s", traceback.format_exc()
            )
            return self._handle_generation_error(str(e))

    def create_system_instruction(self) -> str:
        """Create a concise system instruction for fast listing verification"""
        return """
You are an AI expert in rental property listing verification.
### CORE RULES:
- **REJECT (0.1)**: Cartoons, 3D renders, or watermarks of other sites.
- **NEEDS_REVIEW (0.4-0.6)**: Missing photos, price-location mismatch, or stock photos.
- **APPROVE (0.9-1.0)**: High-quality, realistic photos consistent with the description.
- **INCONSISTENCY**: Flag if the visual view (window) doesn't match the described location.

### RESPONSE FORMAT (JSON ONLY):
{
    "image_validation": {"is_valid": bool, "quality_score": float, "issues": [], "total_images": int, "valid_images": int},
    "video_validation": {"is_valid": bool, "quality_score": float, "issues": [], "total_videos": int, "valid_videos": int},
    "content_validation": {"is_rental_related": bool, "category_match": bool, "content_score": float, "issues": []},
    "completeness_validation": {"is_complete": bool, "completeness_score": float, "missing_fields": [], "quality_issues": []},
    "reason": {"blurriness_issue": bool, "missing_fields": [], "inconsistent_info": bool, "watermark_or_phone": bool, "stock_photo": bool, "details": "string"},
    "violation_codes": ["SCAM", "INAPPROPRIATE_CONTENT", "DUPLICATE_ADS", "WATERMARK_VIOLATION", "INCONSISTENT_INFO", "CONTACT_INFO_IN_DESC"],
    "violations": [{"category": "string", "severity": "low|medium|high|critical", "message": "string"}],
    "suggestions": [{"category": "string", "message": "string", "priority": "low|medium|high"}],
    "is_valid": bool,
    "score": float,
    "confidence": float,
    "suggested_status": "APPROVED|REJECTED|NEEDS_REVIEW"
}
"""

    def create_analysis_prompt(self) -> str:
        return (
            "Please verify this rental listing according to your system instructions."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _response_text(response: Any) -> str:
        """
        Safely extract text from a Vertex AI response.

        `response.text` raises ValueError when the response contains no
        text parts (safety filter, empty candidate, function call).
        Fall back to iterating parts and return empty string if nothing.
        """
        try:
            text = response.text
            return text.strip() if text else ""
        except (ValueError, AttributeError):
            pass
        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    return part.text.strip()
        except (AttributeError, IndexError):
            pass
        return ""

    @staticmethod
    async def _download_images(image_urls: List[str], max_images: int = 8) -> List[Any]:
        """
        Download and preprocess images from URLs concurrently.

        Uses httpx.AsyncClient so we don't block the event loop while waiting
        on network IO, and PIL decoding is offloaded to a worker thread so the
        loop keeps serving other requests during CPU-bound image resizing.
        """
        urls = image_urls[:max_images]
        if not urls:
            return []

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        }
        async with httpx.AsyncClient(headers=headers, timeout=5.0) as client:

            async def _fetch_and_decode(i: int, url: str) -> Any:
                try:
                    resp = await client.get(url)
                    if resp.status_code != 200:
                        logger.warning(
                            "Failed to download image %d: HTTP %s",
                            i + 1,
                            resp.status_code,
                        )
                        return None
                    content = resp.content

                    def _decode() -> Image.Image:
                        img: Image.Image = Image.open(BytesIO(content))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        # Resize to 512px max - This is the most effective way to speed up
                        if img.width > 512 or img.height > 512:
                            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        return img

                    pil_image = await asyncio.to_thread(_decode)
                    logger.info("Successfully processed image %d", i + 1)
                    return pil_image
                except Exception as e:
                    logger.error("Error processing image %d: %s", i + 1, e)
                    return None

            results = await asyncio.gather(
                *(_fetch_and_decode(i, url) for i, url in enumerate(urls))
            )

        return [img for img in results if img is not None]

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
        """Handle and parse API response with sanitization."""
        logger.info("Raw Gemini response: %s...", response_text[:500])
        data = cls._parse_json_response(response_text)

        # Sanitize 'missing_fields' in 'reason' to ensure it's a list (Fixes Pydantic validation error)
        if "reason" in data and isinstance(data["reason"], dict):
            if "missing_fields" in data["reason"]:
                if not isinstance(data["reason"]["missing_fields"], list):
                    logger.warning(
                        "Sanitizing 'reason.missing_fields' from %s to []",
                        type(data["reason"]["missing_fields"]),
                    )
                    data["reason"]["missing_fields"] = []

        # Sanitize 'missing_fields' in 'completeness_validation'
        if "completeness_validation" in data and isinstance(
            data["completeness_validation"], dict
        ):
            if "missing_fields" in data["completeness_validation"]:
                if not isinstance(
                    data["completeness_validation"]["missing_fields"], list
                ):
                    logger.warning(
                        "Sanitizing 'completeness_validation.missing_fields' from %s to []",
                        type(data["completeness_validation"]["missing_fields"]),
                    )
                    data["completeness_validation"]["missing_fields"] = []

        return data

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
