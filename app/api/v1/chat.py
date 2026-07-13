import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.security import require_internal_key
from app.dto.chat import ChatRequest, ChatResponse
from app.service.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service() -> ChatService:
    """Dependency to get chat service instance."""
    try:
        return ChatService()
    except ValueError as e:
        logger.error(f"Chat service initialization failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service not available: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error initializing chat service: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service initialization error: {str(e)}",
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_internal_key)],
)
async def chat(
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """
    Chat with AI assistant to find property listings.

    Send conversation messages and get AI-powered responses.
    The AI can search listings using natural language queries.

    - **messages**: Array of conversation messages with role (user/assistant) and content
    """
    try:
        if not chat_request.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Messages cannot be empty",
            )

        # Validate last message is from user
        if chat_request.messages[-1].role != "user":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Last message must be from user",
            )

        response = await chat_service.process_chat(
            chat_request.messages,
            user_id=chat_request.user_id,
            auth_token=chat_request.auth_token,
            last_listings=chat_request.last_listings,
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error in chat endpoint: {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={"message_count": len(chat_request.messages)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {type(e).__name__}: {str(e)}",
        )


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_internal_key)],
)
async def chat_stream(
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Streaming variant of /chat — returns Server-Sent Events.

    Event types:
        status   — progress signals (thinking, tool_call, tool_result)
        text     — incremental text deltas (`data.delta`)
        listings — listings payload when available
        done     — final metadata, end of stream
        error    — unrecoverable error, stream ends
    """
    if not chat_request.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Messages cannot be empty",
        )
    if chat_request.messages[-1].role != "user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    async def event_generator():
        # SSE comment padding on the first write to defeat kernel-level
        # write batching. 2KB was empirically insufficient on Windows +
        # uvicorn — the kernel held outgoing bytes for ~5 seconds until
        # ~4KB accumulated. 16KB exceeds the threshold and forces immediate
        # flush of headers + padding so the FE's `onopen` fires within
        # ~100ms instead of 5 seconds.
        # Browsers and proxies discard SSE comments per spec (RFC 6202),
        # so this padding is invisible to FE event handlers.
        gen_start = time.perf_counter()
        logger.info("SSE [chat-stream] generator started")
        yield ":" + (" " * 16384) + "\n\n"
        await asyncio.sleep(0)
        logger.info(
            "SSE [chat-stream] padding yielded at +%.0fms",
            (time.perf_counter() - gen_start) * 1000,
        )

        try:
            event_count = 0
            async for event in chat_service.process_chat_stream(
                chat_request.messages,
                user_id=chat_request.user_id,
                auth_token=chat_request.auth_token,
                last_listings=chat_request.last_listings,
            ):
                event_count += 1
                name = event["event"]
                data = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {name}\ndata: {data}\n\n"
                # Yield to the event loop so uvicorn can drain the
                # ASGI send queue to the socket between events.
                await asyncio.sleep(0)
                # Log only the first event's latency (a time-to-first-token
                # proxy). Logging every event put synchronous stdout I/O on
                # the per-token hot path, adding jitter to the token stream.
                if event_count == 1:
                    logger.info(
                        "SSE [chat-stream] first event (%s) at +%.0fms",
                        name,
                        (time.perf_counter() - gen_start) * 1000,
                    )
            logger.info(
                "SSE [chat-stream] completed: %d events in %.0fms",
                event_count,
                (time.perf_counter() - gen_start) * 1000,
            )
        except asyncio.CancelledError:
            logger.info("Client disconnected from /chat/stream")
            raise
        except Exception as e:
            logger.error(
                f"Stream error: {type(e).__name__}: {str(e)}",
                exc_info=True,
                extra={"message_count": len(chat_request.messages)},
            )
            err = json.dumps(
                {"message": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False
            )
            yield f"event: error\ndata: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/health")
async def chat_health() -> dict[str, str]:
    """
    Check if the chat service is healthy.
    """
    try:
        # Try to initialize the service to check if API key is configured
        ChatService()
        return {"status": "healthy", "service": "chat"}
    except ValueError as e:
        logger.error(f"Chat service unavailable: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Chat service unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error(
            f"Error checking chat health: {type(e).__name__}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking chat service health: {type(e).__name__}: {str(e)}",
        )
