"""
Listing-verification helper.

Uses the OpenAI Agents SDK to send a vision-enabled prompt to the configured
LLM provider (Gemini by default via LiteLLM) and parse the JSON output.
The class name is kept for backward compatibility with services that import
`GeminiListingVerificationHelper`; under the hood, the provider is selectable.

Video handling: the OpenAI Responses input format does not have a native
video type. We extract keyframes from each video (via OpenCV in
`app.utils.video_utils.extract_keyframes`) and feed them as additional images.
Videos that fail keyframe extraction are skipped with a warning.
"""

import asyncio
import base64
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, cast

import httpx
from agents import Agent, Runner  # type: ignore[import]
from PIL import Image

from app.ai.llm.agent_factory import default_model_settings, make_model
from app.ai.llm.gateway import get_gateway
from app.core.config import settings

logger = logging.getLogger(__name__)

_MAX_IMAGES = 8


class GeminiListingVerificationHelper:
    """
    Multimodal listing-verification helper. Wraps a single-turn Agents-SDK
    `Agent` (no tools) and parses the JSON response.

    Despite the legacy name, this class is provider-agnostic — see
    `app.ai.llm.agent_factory.make_model`.
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
        """Analyze images and (keyframes from) videos along with text content."""
        try:
            # Download images and videos in parallel
            images_task = self._download_image_data_uris(image_urls)
            videos_task = self._download_video_bytes(video_urls)
            image_uris, video_blobs = await asyncio.gather(images_task, videos_task)

            logger.info(
                "Media download complete: %d images, %d videos loaded.",
                len(image_uris),
                len(video_blobs),
            )

            # Extract keyframes from videos to feed alongside the images
            extra_image_uris = await self._video_keyframes_to_data_uris(video_blobs)
            all_image_uris = image_uris + extra_image_uris
            logger.info(
                "Final payload: %d images (incl. %d keyframes from %d videos)",
                len(all_image_uris),
                len(extra_image_uris),
                len(video_blobs),
            )

            if not all_image_uris:
                logger.warning(
                    "No usable media after processing; falling back to text-only."
                )
                return await self.analyze_text_content(text_content, analysis_prompt)

            full_prompt = (
                "CRITICAL: YOU MUST ANALYZE ALL ATTACHED MEDIA CAREFULLY.\n"
                f"{analysis_prompt}\n\n"
                "### TEXT DATA TO VERIFY:\n"
                f"{text_content}\n\n"
                "### YOUR TASK:\n"
                "1. Cross-reference the provided text with the visual details in the images.\n"
                "2. Check for stock photos, watermarks, and consistency.\n"
                "3. Ensure the media matches the described property.\n"
                "4. Return valid JSON only."
            )

            content_parts: List[Dict[str, Any]] = [
                {"type": "input_text", "text": full_prompt}
            ]
            for uri in all_image_uris:
                content_parts.append(
                    {"type": "input_image", "detail": "auto", "image_url": uri}
                )
            input_items = [{"role": "user", "content": content_parts}]

            trace = self._gateway.create_trace(
                name="listing-verification",
                metadata={
                    "image_count": len(image_uris),
                    "video_count": len(video_blobs),
                    "extra_keyframes": len(extra_image_uris),
                    "mode": "multimodal",
                    "model": settings.LLM_VISION_MODEL,
                },
            )

            response_text = await self._run_one_shot(
                input_items, trace, span_name="verify-multimodal", temperature=0.0
            )
            return self._handle_api_response(response_text)

        except Exception as e:
            logger.error("Multimodal analysis failed: %s", e, exc_info=True)
            return self._handle_generation_error(str(e))

    async def analyze_text_content(
        self, text_content: str, analysis_prompt: str
    ) -> Dict[str, Any]:
        """Analyze text content for listing verification (no images)."""
        try:
            full_prompt = (
                f"{analysis_prompt}\n\n"
                f"Content to analyze:\n{text_content}\n\n"
                "Return your response as valid JSON only, without any additional text or formatting."
            )

            trace = self._gateway.create_trace(
                name="listing-verification",
                metadata={"mode": "text-only", "model": settings.LLM_VISION_MODEL},
            )

            input_items = [{"role": "user", "content": full_prompt}]
            response_text = await self._run_one_shot(
                input_items, trace, span_name="verify-text", temperature=0.0
            )
            return self._handle_api_response(response_text)

        except Exception as e:
            import traceback

            logger.error(
                "[analyze_text_content] FULL ERROR:\n%s", traceback.format_exc()
            )
            return self._handle_generation_error(str(e))

    def create_system_instruction(self) -> str:
        """Concise system instruction for fast listing verification."""
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

    async def _run_one_shot(
        self,
        input_items: List[Dict[str, Any]],
        trace: Any,
        *,
        span_name: str,
        temperature: float,
    ) -> str:
        """Run a one-shot Agent (no tools) and return its final text output."""
        agent = Agent(
            name="Listing Verifier",
            instructions=self.create_system_instruction(),
            model=make_model(settings.LLM_VISION_MODEL),
            model_settings=default_model_settings(temperature=temperature),
        )

        generation = trace.generation(
            name=span_name,
            model=settings.LLM_VISION_MODEL,
            input=str(input_items[0])[:2000],
        )
        try:
            result = await Runner.run(
                starting_agent=agent,
                input=cast(Any, input_items),
                max_turns=2,
            )
            text = str(result.final_output or "")
            generation.end(output=text[:2000])
            return text
        except Exception as e:
            generation.end(level="ERROR", status_message=str(e))
            raise

    @staticmethod
    def _pil_to_data_uri(img: Image.Image, *, max_dim: int = 1024) -> str:
        """Encode a PIL image as a base64 JPEG data URI."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    @classmethod
    async def _download_image_data_uris(cls, image_urls: List[str]) -> List[str]:
        """Download image URLs concurrently and return base64 data URIs."""
        urls = image_urls[:_MAX_IMAGES]
        if not urls:
            return []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:

            async def _fetch(i: int, url: str) -> Optional[str]:
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

                    def _encode() -> str:
                        img: Image.Image = Image.open(BytesIO(content))
                        return cls._pil_to_data_uri(img)

                    return await asyncio.to_thread(_encode)
                except Exception as e:
                    logger.error("Error processing image %d: %s", i + 1, e)
                    return None

            results = await asyncio.gather(
                *(_fetch(i, url) for i, url in enumerate(urls))
            )

        return [u for u in results if u is not None]

    @staticmethod
    async def _download_video_bytes(urls: List[str]) -> List[bytes]:
        """Download videos concurrently. Caps each video at ~10MB."""
        if not urls:
            return []

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
        }

        async with httpx.AsyncClient(headers=headers) as client:

            async def _fetch(i: int, url: str) -> Optional[bytes]:
                try:
                    logger.info("Downloading video %d: %s", i + 1, url)
                    resp = await client.get(
                        url,
                        headers={"Range": "bytes=0-10485760"},
                        timeout=20.0,
                    )
                    if resp.status_code not in (200, 206):
                        resp = await client.get(url, timeout=30.0)
                    if resp.status_code in (200, 206) and len(resp.content) > 0:
                        return resp.content
                except Exception as e:
                    logger.error("Exception downloading video %d: %s", i + 1, e)
                return None

            results = await asyncio.gather(
                *(_fetch(i, url) for i, url in enumerate(urls))
            )

        return [b for b in results if b is not None]

    @classmethod
    async def _video_keyframes_to_data_uris(cls, video_blobs: List[bytes]) -> List[str]:
        """
        Extract keyframes from each video (best-effort) and return them as
        base64 data URIs. Videos that fail extraction are skipped silently
        (with a warning) since the OpenAI Responses input format has no
        native video type.
        """
        if not video_blobs:
            return []

        try:
            from app.utils.video_utils import extract_keyframes
        except ImportError:
            logger.warning(
                "OpenCV not available — videos cannot be analyzed (no keyframe extraction)."
            )
            return []

        all_uris: List[str] = []
        for idx, blob in enumerate(video_blobs):
            try:
                frames = await asyncio.to_thread(extract_keyframes, blob)
            except Exception as e:
                logger.error("Keyframe extraction failed for video %d: %s", idx + 1, e)
                continue

            if not frames:
                logger.warning("No keyframes extracted from video %d", idx + 1)
                continue

            for f in frames:
                try:
                    uri = await asyncio.to_thread(cls._pil_to_data_uri, f)
                    all_uris.append(uri)
                except Exception as e:
                    logger.error(
                        "Failed encoding keyframe from video %d: %s", idx + 1, e
                    )

        return all_uris

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict[str, Any]:
        """Parse JSON response with multiple fallbacks."""
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

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
        """Strip markdown fences, parse JSON, sanitize known shape issues."""
        logger.info("Raw verification response: %s...", response_text[:500])

        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = cls._parse_json_response(text)

        # Sanitize 'missing_fields' in 'reason' to ensure it's a list.
        if "reason" in data and isinstance(data["reason"], dict):
            mf = data["reason"].get("missing_fields")
            if mf is not None and not isinstance(mf, list):
                logger.warning(
                    "Sanitizing 'reason.missing_fields' from %s to []", type(mf)
                )
                data["reason"]["missing_fields"] = []

        if "completeness_validation" in data and isinstance(
            data["completeness_validation"], dict
        ):
            mf = data["completeness_validation"].get("missing_fields")
            if mf is not None and not isinstance(mf, list):
                logger.warning(
                    "Sanitizing 'completeness_validation.missing_fields' from %s to []",
                    type(mf),
                )
                data["completeness_validation"]["missing_fields"] = []

        return data

    @staticmethod
    def _handle_generation_error(error_msg: str) -> Dict[str, Any]:
        """Map errors to a structured payload the service layer understands."""
        logger.error("Error in analysis: %s", error_msg)

        if "quota exceeded" in error_msg.lower() or "429" in error_msg:
            return {"error": "quota exceeded", "analysis_completed": False}
        if "api key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return {"error": "invalid api key", "analysis_completed": False}
        return {"error": error_msg, "analysis_completed": False}
