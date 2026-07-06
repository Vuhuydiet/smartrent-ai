"""
Tool: report_listing — submit a complaint about a listing.

Triggered by "tin này lừa đảo", "báo cáo tin", "nghi tin giả".

Two-step flow:
  1st call (confirm=false, default) → fetch the catalog of valid report
    reasons; the LLM reads it back to the user, who picks one or more.
  2nd call (confirm=true with reasonIds) → tool submits.

The backend requires reporter contact info (reporterEmail + reporterPhone) +
category, so the submit step needs a logged-in user: we auto-fill those from
the user's profile (no need to ask them to type it). Not logged in → we return
a clear "please log in" message rather than letting the backend 400.
"""

import logging
from typing import Annotated, Any, Dict, List, Optional

import httpx
from agents import RunContextWrapper, function_tool  # type: ignore[import]
from pydantic import Field

from app.agent.tool_context import ToolContext
from app.core import backend_client

logger = logging.getLogger(__name__)

# otherFeedback is free-text from the user. Backend will store it and may
# display in admin moderation queues, so cap defensively to keep the
# payload bounded and prevent oversized inputs from running through.
_MAX_OTHER_FEEDBACK_CHARS = 500


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


def _extract_reasons(reasons_data: Any) -> List[Dict[str, Any]]:
    """Normalise the report-reasons payload to a flat list of reason dicts."""
    if isinstance(reasons_data, list):
        return reasons_data
    if isinstance(reasons_data, dict):
        return (
            reasons_data.get("reasons")
            or reasons_data.get("content")
            or reasons_data.get("data")
            or []
        )
    return []


async def _reason_catalog() -> Dict[int, str]:
    """Return {reasonId: reasonText}. Empty on any failure (validation is
    best-effort — never block a report because the catalog re-fetch hiccuped)."""
    try:
        items = _extract_reasons(await backend_client.get_report_reasons())
    except Exception as e:  # noqa: BLE001
        logger.warning("report reason catalog re-fetch failed: %s", e)
        return {}
    catalog: Dict[int, str] = {}
    for it in items:
        rid = it.get("reasonId")
        if rid is not None:
            try:
                catalog[int(rid)] = it.get("reasonText", "")
            except (TypeError, ValueError):
                continue
    return catalog


def _norm_reason(text: str) -> str:
    """Normalise a reasonText for exact matching (lowercase, collapse spaces)."""
    return " ".join(str(text).lower().split())


async def _do_report(
    listing_id: str,
    confirm: bool,
    reason_ids: List[int],
    other_feedback: str,
    token: Optional[str],
    reason_texts: Optional[List[str]] = None,
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

        reasons = _extract_reasons(reasons_data)

        return {
            "status": "needs_confirmation",
            "listingId": listing_id,
            "reasons": reasons,
            "message": (
                "Trình bày các lý do (theo reasonText) cho user chọn. Khi gọi "
                "lại confirm=true, hãy truyền `reasonTexts` = COPY NGUYÊN VĂN "
                "reasonText của (các) lý do user chọn — tool sẽ tự map sang id. "
                "Cách này an toàn nhất; đừng tự đoán reasonId (id không liền "
                "mạch với thứ tự hiển thị)."
            ),
        }

    if len(other_feedback) > _MAX_OTHER_FEEDBACK_CHARS:
        return {
            "status": "error",
            "error": (
                f"otherFeedback quá dài ({len(other_feedback)} ký tự, "
                f"giới hạn {_MAX_OTHER_FEEDBACK_CHARS}). Hãy rút gọn nội "
                "dung phản hồi."
            ),
        }

    # Fetch the catalog once (best-effort): used to map reasonTexts→ids,
    # validate ids, and echo the reported reason on success.
    reason_catalog = await _reason_catalog()

    # Deterministically resolve reasonTexts → reasonIds. The model copies the
    # reasonText verbatim and the tool maps it here, so it never has to pick the
    # non-sequential reasonId (LISTING ids 1-7, MAP ids 8-11, but both restart
    # display_order at 1 → the shown list is interleaved, which is what made the
    # model send the wrong id). Any explicit reasonIds are merged in.
    resolved_ids: List[int] = list(reason_ids)
    if reason_texts and reason_catalog:
        text_to_id = {_norm_reason(t): rid for rid, t in reason_catalog.items()}
        unmatched: List[str] = []
        for t in reason_texts:
            rid = text_to_id.get(_norm_reason(t))
            if rid is None:
                unmatched.append(t)
            elif rid not in resolved_ids:
                resolved_ids.append(rid)
        if unmatched:
            valid = "; ".join(sorted(reason_catalog.values()))
            return {
                "status": "error",
                "error": (
                    f"Không khớp lý do: {unmatched}. Copy đúng reasonText từ "
                    f"danh sách: {valid}"
                ),
            }
    reason_ids = resolved_ids

    if not reason_ids:
        return {
            "status": "error",
            "error": "Cần ít nhất 1 lý do (reasonTexts hoặc reasonIds) khi confirm=true.",
        }

    # Reject any reasonId not in the catalog (a mis-mapped/hallucinated id) so a
    # wrong reason is never submitted to admin.
    if reason_catalog:
        unknown = [i for i in reason_ids if i not in reason_catalog]
        if unknown:
            valid = "; ".join(f"{k}={v}" for k, v in sorted(reason_catalog.items()))
            return {
                "status": "error",
                "error": (
                    f"reasonId không hợp lệ: {unknown}. Hãy dùng đúng reasonId "
                    f"(hoặc truyền reasonTexts) từ danh sách: {valid}"
                ),
            }

    # Backend requires reporterEmail + reporterPhone + category even for
    # "anonymous" (no-auth) reports. Auto-fill contact from the logged-in
    # user's profile so the user doesn't have to type it; if they aren't
    # logged in (or the account lacks contact info), say so clearly instead of
    # letting the backend 400 surface as a vague "system error".
    if not token:
        return {
            "status": "error",
            "error": (
                "Bạn cần đăng nhập để gửi báo cáo — hệ thống cần email và số "
                "điện thoại của người báo cáo."
            ),
        }
    profile = await backend_client.get_user_profile(token)
    if not isinstance(profile, dict) or "error" in profile:
        return {
            "status": "error",
            "error": "Không lấy được thông tin tài khoản để báo cáo. Vui lòng đăng nhập lại.",
        }
    reporter_email = (profile.get("email") or "").strip()
    reporter_phone = (
        profile.get("contactPhoneNumber") or profile.get("phoneNumber") or ""
    ).strip()
    if not reporter_email or not reporter_phone:
        return {
            "status": "error",
            "error": (
                "Tài khoản của bạn chưa có email hoặc số điện thoại — vui lòng "
                "cập nhật hồ sơ trước khi gửi báo cáo."
            ),
        }

    body: Dict[str, Any] = {
        "reasonIds": reason_ids,
        # Backend cross-checks nothing between category and reasons; LISTING is
        # the right default for the chat report flow (valid enum: LISTING|MAP).
        "category": "LISTING",
        "reporterEmail": reporter_email,
        "reporterPhone": reporter_phone,
    }
    if other_feedback:
        body["otherFeedback"] = other_feedback

    try:
        data = await backend_client.submit_listing_report(listing_id, body, token=token)
    except httpx.HTTPStatusError as e:
        # Surface the backend's own message (e.g. "Listing not found",
        # "Report reasons not found") so the model can react correctly instead
        # of telling the user it's a transient system error.
        status = e.response.status_code
        backend_msg = None
        backend_code = None
        try:
            payload = e.response.json()
            backend_msg = payload.get("message")
            backend_code = payload.get("code")
        except Exception:
            backend_msg = (e.response.text or "").strip()[:200] or None
        logger.error(
            "submit_listing_report HTTP %s (code=%s): %s",
            status,
            backend_code,
            backend_msg,
        )
        # Listing no longer publicly visible (backend #348, code 22001) — surface
        # a clear Vietnamese reason instead of the raw English backend message.
        if backend_code == "22001" or (
            status == 400
            and backend_msg
            and "no longer available" in backend_msg.lower()
        ):
            return {
                "status": "error",
                "error": "Tin này không còn hiển thị nên không thể báo cáo.",
            }
        error = f"Báo cáo thất bại (HTTP {status})"
        if backend_msg:
            error += f": {backend_msg}"
        return {"status": "error", "error": error}
    except Exception as e:  # noqa: BLE001
        logger.error("submit_listing_report failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}

    if "error" in data:
        return {"status": "error", "error": data["error"]}

    reported_reasons = [
        reason_catalog.get(i, str(i)) if reason_catalog else str(i) for i in reason_ids
    ]
    return {
        "status": "success",
        "listingId": listing_id,
        "reasonIds": reason_ids,
        # Echo the resolved reason text so the bot confirms to the user exactly
        # what was reported — any id↔reason mismatch is caught immediately.
        "reportedReasons": reported_reasons,
        "message": (
            f"Đã ghi nhận báo cáo tin {listing_id} với lý do: "
            f"{', '.join(str(r) for r in reported_reasons)}. "
            "Cảm ơn bạn đã giúp cộng đồng SmartRent an toàn hơn."
        ),
    }


@function_tool(
    name_override="report_listing",
    description_override=(
        "Report a listing for a violation (scam, fake info, duplicate, "
        "wrong price, inappropriate content). Two-step flow: first call "
        "with confirm=false to fetch the reason catalog and ask the user "
        "which reason applies; then call again with confirm=true. PREFER "
        "passing `reasonTexts` (copy the chosen reasonText verbatim) — the "
        "tool maps text→id safely. `reasonIds` is a fallback but the ids are "
        "NOT sequential with the shown order, so don't guess them. Resolve "
        "listingId from prior conversation context — never ask the user."
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
                "with `reasonTexts` (preferred) or `reasonIds`."
            )
        ),
    ] = None,
    reasonTexts: Annotated[
        Optional[List[str]],
        Field(
            description=(
                "PREFERRED. The reasonText(s) the user picked, copied verbatim "
                "from the catalog. The tool maps each to its reasonId."
            )
        ),
    ] = None,
    reasonIds: Annotated[
        Optional[List[int]],
        Field(
            description=(
                "Fallback: reasonId values from the catalog. Prefer reasonTexts "
                "— do not guess ids (they aren't in display order)."
            )
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
    texts = [str(t) for t in reasonTexts if str(t).strip()] if reasonTexts else None
    return await _do_report(
        listing_id=listing_id,
        confirm=bool(confirm),
        reason_ids=_coerce_reason_ids(reasonIds),
        other_feedback=(otherFeedback or "").strip(),
        token=ctx.context.auth_token,
        reason_texts=texts,
    )
