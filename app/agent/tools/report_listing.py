"""
Tool: report_listing — submit a complaint about a listing.

Triggered by "tin này lừa đảo", "báo cáo tin", "nghi tin giả".

Two-step flow:
  1st call (confirm=false, default) → fetch the catalog of valid report
    reasons; the LLM reads it back to the user, who picks one or more.
  2nd call (confirm=true with reasonIds) → tool submits.

Anonymous reports are allowed by the backend; we forward auth_token when
present so the report is attributed.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)


def _coerce_id(raw: Any) -> str:
    try:
        return str(int(float(raw)))
    except (TypeError, ValueError):
        return str(raw)


def _coerce_reason_ids(raw: Any) -> List[int]:
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for r in raw:
        try:
            out.append(int(float(r)))
        except (TypeError, ValueError):
            continue
    return out


async def _do_report(
    listing_id: str,
    confirm: bool,
    reason_ids: List[int],
    other_feedback: str,
    token: Optional[str],
) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    if not confirm:
        try:
            reasons_data = await backend_client.get_report_reasons()
        except Exception as e:  # noqa: BLE001
            logger.warning("get_report_reasons failed: %s", e)
            return {
                "status": "error",
                "error": "Không lấy được danh sách lý do báo cáo.",
            }
        if isinstance(reasons_data, dict) and "error" in reasons_data:
            return {"status": "error", "error": reasons_data["error"]}

        if isinstance(reasons_data, list):
            reasons = reasons_data
        elif isinstance(reasons_data, dict):
            reasons = (
                reasons_data.get("reasons")
                or reasons_data.get("content")
                or reasons_data.get("data")
                or []
            )
        else:
            reasons = []

        return {
            "status": "needs_confirmation",
            "listingId": listing_id,
            "reasons": reasons,
            "message": (
                "Hãy hỏi user chọn 1 hoặc nhiều lý do từ danh sách trên, "
                "rồi gọi lại tool với confirm=true và reasonIds tương ứng."
            ),
        }

    if not reason_ids:
        return {
            "status": "error",
            "error": "Cần ít nhất 1 reasonId khi confirm=true.",
        }

    body: Dict[str, Any] = {"reasonIds": reason_ids}
    if other_feedback:
        body["otherFeedback"] = other_feedback

    try:
        data = await backend_client.submit_listing_report(listing_id, body, token=token)
    except httpx.HTTPStatusError as e:
        logger.error("submit_listing_report HTTP %s", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend HTTP {e.response.status_code}",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("submit_listing_report failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}

    if "error" in data:
        return {"status": "error", "error": data["error"]}

    return {
        "status": "success",
        "listingId": listing_id,
        "reasonIds": reason_ids,
        "message": (
            f"Đã ghi nhận báo cáo về tin {listing_id}. "
            "Cảm ơn bạn đã giúp cộng đồng SmartRent an toàn hơn."
        ),
    }


@function_tool(
    name_override="report_listing",
    description_override=(
        "Report a listing for a violation (scam, fake info, duplicate, "
        "wrong price, inappropriate content). Two-step flow: first call "
        "with confirm=false to fetch the reason catalog and ask the user "
        "which reason applies; then call again with confirm=true and the "
        "chosen reasonIds. Resolve listingId from prior conversation "
        "context — never ask the user."
    ),
)
async def report_listing(
    ctx: RunContextWrapper[ToolContext],
    listingId: Annotated[
        str, Field(description="Listing ID to report (from context).")
    ],
    confirm: Annotated[
        Optional[bool],
        Field(
            description=(
                "false (default) → return reason catalog. true → submit "
                "with `reasonIds`."
            )
        ),
    ] = None,
    reasonIds: Annotated[
        Optional[List[int]],
        Field(
            description=("Reason IDs the user selected. Required when confirm=true.")
        ),
    ] = None,
    otherFeedback: Annotated[
        Optional[str],
        Field(
            description=(
                "Optional free-text detail from the user about what looked " "off."
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    listing_id = _coerce_id(listingId)
    if not listing_id or listing_id == "None":
        return {"status": "error", "error": "Thiếu listingId."}
    return await _do_report(
        listing_id=listing_id,
        confirm=bool(confirm),
        reason_ids=_coerce_reason_ids(reasonIds),
        other_feedback=(otherFeedback or "").strip(),
        token=ctx.context.auth_token,
    )
