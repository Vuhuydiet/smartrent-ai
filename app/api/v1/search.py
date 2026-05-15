import json
import logging

from agents import Agent, ModelSettings, Runner  # type: ignore[import]

from fastapi import APIRouter, status

from app.ai.llm.agent_factory import make_model
from app.ai.llm.gateway import get_gateway
from app.core.config import settings
from app.dto.search import AiParsedCriteriaDto, SearchParseRequest

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
- keyword: any specific keywords, amenities, or descriptions (e.g. "máy lạnh", "gần đại học", "hẻm xe hơi").
- phoneticKeyword: write the keyword parameter in a way that captures its sound without accents if typo-tolerance is needed (e.g. "may lanh").

If a field is not specified in the query, omit it or set it to null.
"""


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
