"""
Tool: update_listing_price — owner-only. Updates the price of one of the
user's own listings and records the change in pricing history.

Triggered by "hạ giá tin XYZ xuống 5tr", "đổi giá tin của tôi thành 8tr".

Confirm-before-act pattern:
  1st call (confirmed=false, default) → fetch current price for preview;
    return {"status": "needs_confirmation", currentPrice, newPrice, message}.
  2nd call (confirmed=true) → actually apply the update.

The agent only sets confirmed=true after the user has explicitly agreed.
"""

import logging
from typing import Annotated, Any, Dict, Optional

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


def _coerce_price(raw: Any) -> Optional[float]:
    """Accept '5tr' / '5_000_000' / 5000000 / 5e6. Return None on failure."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower().replace(",", "").replace("_", "").replace(" ", "")
    multiplier = 1.0
    if s.endswith("tr") or s.endswith("triệu") or s.endswith("trieu"):
        multiplier = 1_000_000.0
        s = s.rstrip("triệutrieu")
    elif s.endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return None


async def _do_update_price(
    token: str, listing_id: str, new_price: float, confirmed: bool
) -> Dict[str, Any]:
    """Core logic — separated so it can be called directly in tests."""
    if not confirmed:
        raw_price: Any = None
        try:
            current = await backend_client.get_listing(listing_id)
            if isinstance(current, dict):
                raw_price = current.get("price")
        except Exception:  # noqa: BLE001
            # The public detail endpoint returns 404 for listings that aren't
            # currently visible (pending review, rejected, expired, suspended —
            # backend #352). The owner can still reprice via the owner-scoped
            # endpoint; we just can't preview the old price here.
            raw_price = None
        old_price = raw_price if isinstance(raw_price, (int, float)) else None
        if old_price is not None:
            message = (
                f"Xác nhận đổi giá tin {listing_id} từ {int(old_price):,} sang "
                f"{int(new_price):,} VND? (Trả lời 'có' để xác nhận.)"
            )
        else:
            message = (
                f"Xác nhận đổi giá tin {listing_id} thành {int(new_price):,} VND? "
                "(Chưa lấy được giá hiện tại — có thể tin đang chờ duyệt hoặc đã "
                "hết hạn. Trả lời 'có' để xác nhận.)"
            )
        return {
            "status": "needs_confirmation",
            "listingId": listing_id,
            "currentPrice": old_price,
            "newPrice": new_price,
            "message": message,
        }

    try:
        data = await backend_client.update_listing_price(
            listing_id, new_price, token=token
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            return {
                "status": "error",
                "error": "Bạn không có quyền sửa tin này.",
            }
        if e.response.status_code == 404:
            return {
                "status": "error",
                "error": f"Không tìm thấy tin {listing_id}.",
            }
        logger.error("update_listing_price HTTP %s", e.response.status_code)
        return {
            "status": "error",
            "error": f"Backend returned HTTP {e.response.status_code}",
        }
    except Exception as e:  # noqa: BLE001
        logger.error("update_listing_price failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}

    if "error" in data:
        return {"status": "error", "error": data["error"]}

    return {
        "status": "success",
        "listingId": listing_id,
        "newPrice": new_price,
        "message": f"Đã cập nhật giá tin {listing_id} thành {int(new_price):,} VND.",
    }


@function_tool(
    name_override="update_listing_price",
    description_override=(
        "Update the price of ONE of the user's OWN listings. Triggers: 'hạ "
        "giá tin X xuống 5tr', 'đổi giá thành 8 triệu'. Confirm-before-act: "
        "set confirmed=true ONLY after the user has explicitly agreed to "
        "the new price. Without confirmation, the tool returns a preview "
        "and asks for confirmation."
    ),
)
async def update_listing_price(
    ctx: RunContextWrapper[ToolContext],
    listingId: Annotated[
        str, Field(description="The user's own listing ID to reprice.")
    ],
    newPrice: Annotated[
        float,
        Field(
            description=(
                "New price. Accepts plain VND (5000000), shorthand strings "
                "('5tr' / '5_000_000' / '5,000,000'), or scientific (5e6). "
                "All are normalised to a positive float internally."
            )
        ),
    ],
    confirmed: Annotated[
        Optional[bool],
        Field(
            description=(
                "Set to true ONLY after the user has explicitly agreed to "
                "the change. Defaults to false (preview)."
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    token = ctx.context.auth_token
    if not token:
        return {
            "status": "error",
            "error": "Người dùng chưa đăng nhập. Hãy hướng dẫn đăng nhập trước.",
        }

    listing_id = _coerce_id(listingId)
    price = _coerce_price(newPrice)
    if not listing_id or listing_id == "None":
        return {"status": "error", "error": "Thiếu listingId."}
    if price is None or price <= 0:
        return {
            "status": "error",
            "error": "Giá mới không hợp lệ. Hãy hỏi user lại giá cụ thể.",
        }

    return await _do_update_price(token, listing_id, price, bool(confirmed))
