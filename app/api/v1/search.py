import json
import logging
import re
import unicodedata

from agents import Agent, ModelSettings, Runner  # type: ignore[import]

from fastapi import APIRouter, status

from app.ai.llm.agent_factory import make_model
from app.ai.llm.gateway import get_gateway
from app.core.config import settings
from app.dto.search import (
    AiParsedCriteriaDto,
    SearchParseRequest,
    SearchSuggestionRequest,
    SearchSuggestionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_INSTRUCTION = """
You are a real estate search intent parser.
Your task is to parse a Vietnamese natural language search query and extract structured criteria.
Only return a JSON object that matches the provided schema.

Rules:
- propertyType: must be one of APARTMENT, HOUSE, ROOM, STUDIO, OFFICE. (e.g. "phòng trọ" -> ROOM, "căn hộ" -> APARTMENT, "nhà" -> HOUSE).
- listingType: must be one of RENT, SALE, SHARE. If missing, assume RENT.
- minPrice / maxPrice: extract the price in VND. (e.g. "dưới 5 triệu" -> maxPrice: 5000000. "từ 2 đến 3 triệu" -> minPrice: 2000000, maxPrice: 3000000).
- province / district / ward: extract the location explicitly. If "quận 1" is found, set district="1". If "hồ chí minh" is found, set province="Hồ Chí Minh".
- amenities: extract concrete amenities as Vietnamese text, e.g. ["máy lạnh", "full nội thất"].
- keyword: remaining descriptive keywords that are not already structured filters or amenities, e.g. "gần đại học", "hẻm xe hơi".
- phoneticKeyword: write the keyword parameter in a way that captures its sound without accents if typo-tolerance is needed (e.g. "may lanh").

If a field is not specified in the query, omit it or set it to null.
"""

SUGGESTION_SYSTEM_INSTRUCTION = """
You normalize short Vietnamese rental search fragments into complete user-facing search suggestions.
Return only JSON with a "suggestions" array.

Rules:
- Do not search listings and do not generate SQL.
- Expand abbreviations: q1 -> quận 1, dhqg -> đại học quốc gia, 5tr -> 5 triệu.
- Keep suggestions short, natural, and suitable for autocomplete.
- Prefer rental intent if the user does not specify sale/share.
- Use Vietnamese accents in output.
"""

LOCAL_SUGGESTIONS = [
    "phòng trọ quận 1 dưới 5 triệu",
    "căn hộ quận 1 giá dưới 5 triệu",
    "căn hộ gần đại học quốc gia",
    "căn hộ full nội thất gần đại học quốc gia",
    "phòng trọ có máy lạnh",
    "phòng trọ full nội thất có máy lạnh",
    "phòng full nội thất bình thạnh",
    "nhà trọ giá rẻ quận 7",
    "phòng trọ quận 7 giá rẻ",
    "căn hộ bình thạnh full nội thất",
    "studio gần trung tâm",
    "phòng trọ thủ đức gần đại học quốc gia",
]


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"[^a-zA-Z0-9\s]", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def _normalize_intent(value: str) -> str:
    result = f" {_normalize_text(value)} "
    replacements = {
        " canho ": " can ho ",
        " phongtro ": " phong tro ",
        " nha tro ": " phong tro ",
        " tro ": " phong tro ",
        " dhqg ": " dai hoc quoc gia ",
        " may lan ": " may lanh ",
        " maylanh ": " may lanh ",
        " full nt ": " full noi that ",
    }
    for source, target in replacements.items():
        result = result.replace(source, target)
    for district in range(1, 13):
        result = result.replace(f" q{district} ", f" quan {district} ")
    result = re.sub(r"\b(\d{1,3})\s*(tr|trieu)\b", r"\1 trieu", result)
    return re.sub(r"\s+", " ", result).strip()


def _local_suggestions(query: str, limit: int) -> list[str]:
    normalized = _normalize_intent(query)
    tokens = [token for token in normalized.split() if len(token) >= 2]
    if not tokens:
        return []

    ranked: list[tuple[int, str]] = []
    for suggestion in LOCAL_SUGGESTIONS:
        suggestion_norm = _normalize_text(suggestion)
        matches = sum(1 for token in tokens if token in suggestion_norm)
        if normalized in suggestion_norm:
            matches += 3
        if matches:
            ranked.append((matches, suggestion))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in ranked[:limit]]


@router.post(
    "/parse", response_model=AiParsedCriteriaDto, status_code=status.HTTP_200_OK
)
async def parse_search_query(request: SearchParseRequest) -> AiParsedCriteriaDto:
    if not request.query or not request.query.strip():
        return AiParsedCriteriaDto()

    gateway = get_gateway()
    model_name = settings.LLM_CHAT_MODEL

    # Define JSON schema for Vertex AI
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "propertyType": {"type": "STRING"},
            "listingType": {"type": "STRING"},
            "minPrice": {"type": "NUMBER"},
            "maxPrice": {"type": "NUMBER"},
            "province": {"type": "STRING"},
            "district": {"type": "STRING"},
            "ward": {"type": "STRING"},
            "amenities": {"type": "ARRAY", "items": {"type": "STRING"}},
            "keyword": {"type": "STRING"},
            "phoneticKeyword": {"type": "STRING"},
        },
    }

    trace = gateway.create_trace(
        name="search-parse",
        input=request.query,
        metadata={"model": model_name},
    )

    try:
        agent = Agent(
            name="Search Parser",
            instructions=SYSTEM_INSTRUCTION,
            model=make_model(model_name),
            model_settings=ModelSettings(
                temperature=0.1,
                extra_body={
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": response_schema,
                    }
                },
            ),
        )

        span = trace.generation(
            name="search-parse-generate",
            model=model_name,
            input=request.query,
        )

        try:
            run_result = await Runner.run(
                starting_agent=agent,
                input=f"Parse this query: '{request.query}'",
                max_turns=2,
            )
            text = str(run_result.final_output or "")
            span.end(output=text[:2000])
        except Exception as e:
            span.end(level="ERROR", status_message=str(e))
            raise

        if not text:
            return AiParsedCriteriaDto()

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("\n```", 1)[0]
            text = text.strip()

        parsed_data = json.loads(text)
        trace.update(output={"parsed": parsed_data})

        return AiParsedCriteriaDto(**parsed_data)

    except Exception as e:
        logger.error("Error calling search parse endpoint: %s", e, exc_info=True)
        trace.update(output={"error": str(e)})
        # Instead of failing the user request, return empty criteria so the backend falls back to FULLTEXT search.
        return AiParsedCriteriaDto(keyword=request.query)


@router.post(
    "/suggestions",
    response_model=SearchSuggestionResponse,
    status_code=status.HTTP_200_OK,
)
async def suggest_search_queries(
    request: SearchSuggestionRequest,
) -> SearchSuggestionResponse:
    if not request.query or len(request.query.strip()) < 3:
        return SearchSuggestionResponse(suggestions=[], normalizedQuery="")

    safe_limit = max(1, min(request.limit or 5, 8))
    normalized = _normalize_intent(request.query)
    local = _local_suggestions(request.query, safe_limit)

    if len(local) >= min(3, safe_limit):
        return SearchSuggestionResponse(
            suggestions=local[:safe_limit],
            normalizedQuery=normalized,
        )

    gateway = get_gateway()
    model_name = settings.LLM_CHAT_MODEL
    trace = gateway.create_trace(
        name="search-suggestions",
        input=request.query,
        metadata={"model": model_name, "normalized": normalized},
    )

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "suggestions": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            }
        },
        "required": ["suggestions"],
    }

    try:
        agent = Agent(
            name="Search Suggestion Normalizer",
            instructions=SUGGESTION_SYSTEM_INSTRUCTION,
            model=make_model(model_name),
            model_settings=ModelSettings(
                temperature=0.2,
                extra_body={
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "responseSchema": response_schema,
                    }
                },
            ),
        )

        span = trace.generation(
            name="search-suggestions-generate",
            model=model_name,
            input=request.query,
        )
        try:
            run_result = await Runner.run(
                starting_agent=agent,
                input=(
                    f"Input fragment: '{request.query}'. "
                    f"Normalized fragment: '{normalized}'. "
                    f"Return at most {safe_limit} suggestions."
                ),
                max_turns=2,
            )
            text = str(run_result.final_output or "")
            span.end(output=text[:2000])
        except Exception as e:
            span.end(level="ERROR", status_message=str(e))
            raise

        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text.rsplit("\n```", 1)[0]
            text = text.strip()

        parsed = json.loads(text or "{}")
        ai_suggestions = parsed.get("suggestions") or []
        merged = []
        for suggestion in [*local, *ai_suggestions]:
            if not isinstance(suggestion, str) or not suggestion.strip():
                continue
            if suggestion not in merged:
                merged.append(suggestion)
            if len(merged) >= safe_limit:
                break

        trace.update(output={"suggestions": merged})
        return SearchSuggestionResponse(
            suggestions=merged,
            normalizedQuery=normalized,
        )
    except Exception as e:
        logger.error("Error calling search suggestions endpoint: %s", e, exc_info=True)
        trace.update(output={"error": str(e)})
        return SearchSuggestionResponse(
            suggestions=local[:safe_limit],
            normalizedQuery=normalized,
        )
