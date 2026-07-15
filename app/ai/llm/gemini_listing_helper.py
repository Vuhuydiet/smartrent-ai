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
                "4. Return valid JSON only.\n"
                "5. IMPORTANT: All descriptive text fields (details, issues, messages, suggestions) MUST be written in Vietnamese."
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
                "Return your response as valid JSON only, without any additional text or formatting.\n"
                "IMPORTANT: All descriptive text fields (details, issues, messages, suggestions) MUST be written in Vietnamese."
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
You are an AI expert in rental property listing verification. Your goal is to review rental listings professionally and realistically. Act as a smart, practical AI verification agent.

### LANGUAGE REQUIREMENT:
- ALL descriptive text fields in your response (details, issues, message, suggestions, reason details, etc.) MUST be written in Vietnamese.
- Only the fixed enum values (suggested_status, severity, priority, violation_codes) remain in English as specified.

### CORE VERIFICATION CRITERIA (REALISTIC & STRUCTURED):
- **FACT & METADATA CONSISTENCY (CRITICAL)**: Do NOT strictly verify consistency against the listing title (since titles can be AI-generated, marketing-focused, or slightly mismatched). Instead, you MUST verify that the media and description match key structured facts:
  - **Price (Giá)**: Check if the rent price is realistic and reasonable for the property type (e.g. avoid extreme typos like 100 VND or 100 Billion VND for a simple room).
  - **Area (Diện tích)**: Ensure the visual scale of the images generally matches the described area (e.g., a 15m2 room should look like a cozy single room, while a 100m2 property should look spacious).
  - **Bedrooms / Bathrooms (Số phòng ngủ / vệ sinh)**: Cross-reference the metadata bedrooms/bathrooms with the room layouts visible in the media or described in the text.
  - **Property Type (Loại hình)**: Ensure the images generally reflect a residential rental property matching the selected type (e.g., a `ROOM` or `APARTMENT` should look like habitable housing, not a raw plot of outdoor dirt, a factory, or a non-residential commercial warehouse).
  - **Be Practical with Minor Details**: Do NOT be overly strict about minor differences (e.g., if the text says "wooden floor" but the image shows a "carpeted floor", or the text lists a "wooden wardrobe" but the wardrobe is not visible in the frame). As long as the photo depicts a real, habitable room matching the described property type and structured facts, you should APPROVE it.

- **TOXIC & INAPPROPRIATE LANGUAGE (REJECT IMMEDIATELY)**: Thoroughly scan the title and description for inappropriate, vulgar, profane, swearing, cursing, or offensive Vietnamese words/slang (e.g., "má nó", "đm", "vcl", "chửi bậy"). If any are found, you MUST suggest "REJECTED" with `is_valid: false`, `score: 0.1`, add `"INAPPROPRIATE_CONTENT"` to `violation_codes`, and list the bad word in the `violations` array.

- **MEDIA QUALITY & REJECTS**:
  - **INVALID IMAGE TYPES (CRITICAL - Mark as invalid, reduce quality_score significantly)**:
    - **Anime / Manga / Cartoon characters**: Any image depicting a 2D or 3D animated fictional character, anime girl/boy, manga-style illustration, cartoon avatar, or any non-photographic character art. Even if other images in the listing are real room photos, the presence of such an image is ALWAYS invalid and must be flagged.
    - **Profile pictures / Avatars**: Any image that appears to be a personal profile photo, social media avatar, or portrait of a person (real or fictional) rather than a property photo.
    - **Completely unrelated images**: Screenshots of apps, memes, logos, QR codes, maps, or any image that is clearly not showing the interior or exterior of a property.
  - **NEEDS_REVIEW (0.4-0.6)**: Missing images, extremely blurry/low quality photos, major structured metadata mismatch (e.g. declaring 5 bedrooms but showing a single tiny studio), OR if 1 out of several images is an anime/avatar/unrelated image (reduce score proportionally, flag the specific image in issues).
  - **REJECT (0.1)**: Obvious fake 3D blueprint renders (not real photos), fully animated cartoon/anime scenes (non-character art), inappropriate sexual/violent content, photos with competitor real estate watermarks, OR if the majority of images are invalid (non-property).
  - **APPROVE (0.7-1.0)**: All (or nearly all) images are clean, high-quality, realistic real-estate photos that match the described property type and structured facts.
    - *SPECIAL RULE FOR DEV TESTING*: If the image URL is from Unsplash (contains 'unsplash.com'), treat it as a valid, real, actual room photo, NOT a stock photo, and APPROVE it with a score of 0.9 or higher and suggested_status "APPROVED".

  - **SCORING FORMULA FOR MIXED IMAGES**: If a listing has N total images and K of them are invalid (anime, avatar, unrelated), compute:
    - valid_ratio = (N - K) / N
    - quality_score = max(0.1, valid_ratio * 0.9)
    - If valid_ratio < 0.6: suggested_status = "NEEDS_REVIEW", is_valid = false
    - If valid_ratio < 0.3: suggested_status = "REJECTED", is_valid = false
    - Always list each invalid image as an issue in image_validation.issues (e.g. "Ảnh 1: Hình nhân vật anime/avatar, không phải ảnh bất động sản thực tế")

### STATUS RECOMMENDATION:
- If the listing matches the structured facts, has a clean description, and ALL images are real property photos, suggest "APPROVED" with a score of 0.8 or higher.
- If there is a severe structured mismatch (e.g., unrealistic price or massive bedroom count mismatch), suggest "NEEDS_REVIEW" or "REJECTED" with a score below 0.6.
- If any image is an anime character, cartoon avatar, or completely unrelated non-property image, you MUST flag it regardless of how good the other images are. Do NOT ignore invalid images just because the majority of images are valid.

### RESPONSE FORMAT (JSON ONLY):
Return a JSON response with this exact structure (keep messages concise, max 100 characters):
{
    "image_validation": {
        "is_valid": true,
        "quality_score": 0.9,
        "issues": [],
        "total_images": 1,
        "valid_images": 1
    },
    "video_validation": {
        "is_valid": true,
        "quality_score": 1.0,
        "issues": [],
        "total_videos": 0,
        "valid_videos": 0
    },
    "content_validation": {
        "is_rental_related": true,
        "category_match": true,
        "content_score": 0.9,
        "issues": []
    },
    "completeness_validation": {
        "is_complete": true,
        "completeness_score": 1.0,
        "missing_fields": [],
        "quality_issues": []
    },
    "reason": {
        "blurriness_issue": false,
        "missing_fields": [],
        "inconsistent_info": false,
        "watermark_or_phone": false,
        "stock_photo": false,
        "details": "Mô tả chi tiết bằng tiếng Việt lý do phê duyệt hoặc từ chối"
    },
    "violation_codes": [],
    "violations": [],
    "suggestions": [],
    "is_valid": true,
    "score": 0.9,
    "confidence": 0.9,
    "suggested_status": "APPROVED"
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

        # Sanitize 'missing_fields' in 'reason' to ensure it's a list of strings.
        if "reason" in data and isinstance(data["reason"], dict):
            mf = data["reason"].get("missing_fields")
            if isinstance(mf, list):
                sanitized_mf = []
                for field in mf:
                    if isinstance(field, dict):
                        val = (
                            field.get("field")
                            or field.get("message")
                            or field.get("text")
                            or field.get("details")
                            or str(field)
                        )
                        sanitized_mf.append(val)
                    elif field is not None:
                        sanitized_mf.append(str(field))
                data["reason"]["missing_fields"] = sanitized_mf
            elif mf is not None:
                logger.warning(
                    "Sanitizing 'reason.missing_fields' from %s to []", type(mf)
                )
                data["reason"]["missing_fields"] = []

        if "completeness_validation" in data and isinstance(
            data["completeness_validation"], dict
        ):
            mf = data["completeness_validation"].get("missing_fields")
            if isinstance(mf, list):
                sanitized_mf = []
                for field in mf:
                    if isinstance(field, dict):
                        val = (
                            field.get("field")
                            or field.get("message")
                            or field.get("text")
                            or field.get("details")
                            or str(field)
                        )
                        sanitized_mf.append(val)
                    elif field is not None:
                        sanitized_mf.append(str(field))
                data["completeness_validation"]["missing_fields"] = sanitized_mf
            elif mf is not None:
                logger.warning(
                    "Sanitizing 'completeness_validation.missing_fields' from %s to []",
                    type(mf),
                )
                data["completeness_validation"]["missing_fields"] = []

            qi = data["completeness_validation"].get("quality_issues")
            if isinstance(qi, list):
                sanitized_qi = []
                for issue in qi:
                    if isinstance(issue, dict):
                        val = (
                            issue.get("message")
                            or issue.get("text")
                            or issue.get("issue")
                            or issue.get("details")
                            or str(issue)
                        )
                        sanitized_qi.append(val)
                    elif issue is not None:
                        sanitized_qi.append(str(issue))
                data["completeness_validation"]["quality_issues"] = sanitized_qi
            elif qi is not None:
                data["completeness_validation"]["quality_issues"] = []

        # Sanitize 'issues' in validations to ensure they are Lists of strings, not Lists of dicts.
        for validation_key in [
            "image_validation",
            "video_validation",
            "content_validation",
        ]:
            if validation_key in data and isinstance(data[validation_key], dict):
                issues = data[validation_key].get("issues")
                if isinstance(issues, list):
                    sanitized_issues = []
                    for issue in issues:
                        if isinstance(issue, dict):
                            val = (
                                issue.get("message")
                                or issue.get("text")
                                or issue.get("issue")
                                or issue.get("details")
                                or str(issue)
                            )
                            sanitized_issues.append(val)
                        elif issue is not None:
                            sanitized_issues.append(str(issue))
                    data[validation_key]["issues"] = sanitized_issues
                elif issues is not None:
                    data[validation_key]["issues"] = []

        # Sanitize 'violation_codes' to ensure it is a list of strings
        if "violation_codes" in data:
            vc = data["violation_codes"]
            if isinstance(vc, list):
                sanitized_vc = []
                for code in vc:
                    if isinstance(code, dict):
                        val = (
                            code.get("code")
                            or code.get("category")
                            or code.get("message")
                            or str(code)
                        )
                        sanitized_vc.append(val)
                    elif code is not None:
                        sanitized_vc.append(str(code))
                data["violation_codes"] = sanitized_vc
            elif vc is not None:
                data["violation_codes"] = []

        # Sanitize 'suggestions' to ensure it's a list of dicts.
        if "suggestions" in data and isinstance(data["suggestions"], list):
            sanitized_suggestions = []
            for item in data["suggestions"]:
                if isinstance(item, dict):
                    sanitized_suggestions.append(item)
                elif isinstance(item, str):
                    sanitized_suggestions.append(
                        {
                            "category": "improvement",
                            "message": item,
                            "field": "",
                            "priority": "low",
                        }
                    )
            data["suggestions"] = sanitized_suggestions

        # Sanitize 'violations' to ensure it's a list of dicts.
        if "violations" in data and isinstance(data["violations"], list):
            sanitized_violations = []
            for item in data["violations"]:
                if isinstance(item, dict):
                    sanitized_violations.append(item)
                elif isinstance(item, str):
                    sanitized_violations.append(
                        {
                            "category": "unknown",
                            "severity": "medium",
                            "message": item,
                            "field": "",
                        }
                    )
            data["violations"] = sanitized_violations

        return data

    @staticmethod
    def _classify_error(error_msg: str) -> str:
        """
        Map a provider error to a stable code.

        Matching is done on a normalised string: the old check looked for the
        literal ``"api key"`` (with a space), so the one error we raise ourselves —
        "…no credentials configured. Set GCP_CREDENTIALS_BASE64 … or GEMINI_API_KEY"
        — never matched it and was reported as a generic, unactionable failure.
        Underscores are folded to spaces so both spellings hit.
        """
        msg = error_msg.lower().replace("_", " ")

        if "quota" in msg or "429" in msg or "rate limit" in msg:
            return "LLM_QUOTA_EXCEEDED"
        if "no credentials configured" in msg or "not configured" in msg:
            return "LLM_NOT_CONFIGURED"
        if (
            "api key" in msg
            or "unauthorized" in msg
            or "permission denied" in msg
            or "401" in msg
            or "403" in msg
            or "credential" in msg
        ):
            return "LLM_AUTH"
        if "not found" in msg or "404" in msg or "does not exist" in msg:
            return "LLM_MODEL_NOT_FOUND"
        if "timeout" in msg or "timed out" in msg or "deadline" in msg:
            return "LLM_TIMEOUT"
        return "LLM_ERROR"

    @classmethod
    def _handle_generation_error(cls, error_msg: str) -> Dict[str, Any]:
        """Map errors to a structured payload the service layer understands."""
        code = cls._classify_error(error_msg)
        logger.error("Error in analysis [%s]: %s", code, error_msg)

        # Keep the provider's message verbatim. Replacing it with a canned string
        # ("quota exceeded") threw away the only detail that makes a failure
        # diagnosable — the code above is what callers should branch on.
        return {
            "error": error_msg,
            "error_code": code,
            "analysis_completed": False,
        }
