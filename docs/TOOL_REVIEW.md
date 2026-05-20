# Tool Review — Sprint v2 + Pre-existing Tools

**Date**: 2026-05-11
**Branch**: `chore/tool-review-week1`
**Scope**: 14 `@function_tool` callables under `app/agent/tools/`
**Goal**: Find logic bugs, edge cases, validation gaps, and consistency
issues before thesis defense (2026-07-10). Not a refactor — small fixes
only.

## Severity legend
- 🔴 Bug — wrong behavior or runtime risk
- 🟡 Edge case / UX awkwardness
- 🟢 Cosmetic / consistency

---

## Per-tool findings

### compare_listings
- 🟡 On total-fail (all listings unreachable), success payload lacks
  `"listings": []` field — API consumer must branch on missing key.
  Suggest: always include the field.
- 🟢 description_override is 231 chars (target ≤150).

### my_listings_status
- 🟡 `focus` enum includes `"all"` but `_do_my_listings_status` treats
  None equivalently via `focus or "all"`. Public surface accepts both
  None and `"all"`; pick one. Suggest: drop `"all"` from enum, document
  None = full summary.
- 🟢 description_override is 178 chars (mild overshoot).

### address_translator 🔴 **fixed in this branch**
- 🔴 **Substring match too loose** — `qn in _normalise(candidate)`
  accepts any partial overlap. Verified with reproducer:
  ```
  '1'    → Quận 1     (false positive — "1" silently picks the first district whose normalised alias contains "1")
  'q'    → Quận 1     (false positive)
  'Đông' → Đống Đa    (could mean Đông Anh, Thủ Đức, Bắc Từ Liêm)
  'Từ Liêm' → Nam Từ Liêm  (could be Bắc Từ Liêm)
  ```
  Fix: two-pass match — exact (whole-string) first; substring fallback
  only when `len(qn) >= 4`. See commit on this branch.
- 🟡 No log when `_normalise(query)` returns empty (whitespace-only or
  non-Vietnamese chars). Silent None return is hard to debug.
- 🟢 description_override is 316 chars.

### bulk_save_listings
- 🟡 Silent truncation when caller passes >10 ids. Suggest adding
  `truncated: bool` flag in response or mentioning in message.
- 🟢 Description is 189 chars.

### update_listing_price 🟡 **fixed in this branch**
- 🟡 `newPrice` parameter description says "Always pass the full numeric
  VND amount" but `_coerce_price` accepts `"5tr"`, `"5_000_000"`, etc.
  Mismatch — agent might write extra logic to format. Fix: update field
  description to reflect actual coercion.
- 🟢 Otherwise clean.

### notifications_inbox
- 🟡 Race between `read` and `isRead` fields in backend payload. If
  backend returns both with conflicting values, OR-logic gives wrong
  unread count. Low likelihood (backend uses one or the other), defer.
- 🟢 description_override is 223 chars.

### report_listing 🟡 **fixed in this branch**
- 🟡 `otherFeedback` accepted with no length cap. Backend may reject
  10MB. Fix: cap at 500 chars in tool, return error if exceeded.
- 🟢 The `needs_confirmation` `message` field contains an instruction
  *for the LLM caller* ("Hãy hỏi user..."). This is correct intent
  but confusing convention — message reads like user-facing text. Not
  changing wording (LLM consumes it correctly), but document.

### search_listings
- 🟡 **Multi-type pagination quirk**: when `productTypes` >= 2 and
  `page > 1`, each type's call passes the same `page` value, then
  results merge + truncate to `size`. User asking page 2 may see
  page-1 listings from another type. Decision: document, don't fix
  — fixing requires per-type cursor tracking.
- 🟡 `_MAX_PRODUCT_TYPES = 5` silently drops excess. Suggest log
  warning when cap is hit.
- 🟡 If user passes BOTH `provinceCode="01"` and `provinceId="02"`
  (conflicting), no validation. Both go to backend. Suggest: log
  warning, prefer `provinceCode`.
- 🟢 description_override is 168 chars.

### get_listing_detail
- 🟡 `int(float(listingId))` accepts any number including negative or
  10^15. Defer — backend will 404, error message acceptable.
- 🟢 Description is 165 chars.

### get_price_estimate
- 🔴 **Division-by-zero risk** in rule-based fallback (line 251):
  `diff_pct = (askingPrice - mid) / mid * 100`. If `mid == 0`
  (all multipliers zero, unlikely but possible), crashes. Defer:
  current data shape makes this practically impossible, but add
  guard if we touch this file again.
- 🟡 No range validation on `latitude`/`longitude`. Backend
  presumably validates. Defer.

### get_price_history 🔴 **fixed in this branch**
- 🔴 **Uncaught ValueError on non-numeric listingId** — `_normalize_listing_id`
  returns the raw string if int conversion fails (lines 38-41), then
  `_get_history` calls `int(listing_id)` (line 50) without a try/except.
  User typing `listingId="abc"` crashes the tool. Fix: validate
  numeric before dispatching.
- 🟡 `ADJUSTED` entries filtered in `_get_history` but not in
  `_get_statistics` — counts misalign. Defer (no user-visible
  impact yet).
- 🟡 `daysBack` clamped to [1, 365] silently.

### get_recommendations
- 🟡 `_personalized()` checks auth, `_similar()` doesn't. If backend
  rejects unauthenticated `/similar/{id}` request, error bubbles up;
  if it allows, behavior differs. Defer — backend contract OK as-is.
- 🟡 `topN` clamped to [1, 20] silently.

### get_user_info
- 🟡 `_get_membership` returns `status:success` with `active:false`
  when backend errors. Semantically reasonable but blurs "no
  membership" vs "backend error". Defer.

### save_listing
- 🟢 Reference quality. No findings.

---

## Cross-cutting concerns

### 1. Description length policy
Several tools overshoot a soft 150-char target. The wording was
optimised for clarity, but description tokens are paid every round.
Decision: **defer trim** — current latency is acceptable (~5s warm),
and shortening risks ambiguity. Re-visit if warm latency creeps over
8s after the 200k-listing scale-up.

### 2. ID normalisation inconsistency
Three tools use `int(float(x))` (save_listing, get_listing_detail,
update_listing_price). One uses a custom helper that returns the raw
string on failure (get_price_history). The custom helper hides errors.
Decision: keep per-tool normalisation, document the convention. Don't
introduce a shared helper just for this — would force a refactor.

### 3. Silent parameter clamping
`size`, `topN`, `daysBack`, `productTypes` cap, `bulk_save_listings`
ids cap — all clamp silently. The LLM doesn't know it lost data.
Pattern fix: log at INFO when clamped. Not done in this branch.

### 4. Token guard placement
Most authed tools check `ctx.context.auth_token` inside the public
function before delegating. `save_listing` factors the guard into a
private `_handle_save` for testability. Both work; not standardising
in this pass.

### 5. Backend response schema assumptions
Tools assume `data.get("error")` reliably surfaces failures. None
validate response shape. Malformed/empty responses fall through to
silent success with empty data. Not a current bug but worth a
defensive `try/except` audit if we add new tools.

---

## Fixes applied in this branch

1. **address_translator**: two-pass matching (exact whole-string first,
   substring fallback when query length ≥ 4). Fixes the critical
   ambiguous-input bug.
2. **report_listing**: `otherFeedback` capped at 500 chars; rejection
   message returned if exceeded.
3. **get_price_history**: validate `listingId` is numeric before
   `int()` cast in `_get_history`; return structured error if not.
4. **update_listing_price**: `newPrice` field description updated to
   document `"5tr"` / `"5_000_000"` / numeric all accepted.

## Deferred (documented only)

- Description trim across 4 tools (compare/my_listings/address/notifications)
- `my_listings_status.focus` enum redesign
- `search_listings` multi-type pagination per-type cursor
- `get_price_estimate` div-by-zero guard
- Silent clamp logging (size/topN/daysBack)

## Re-enable test in follow-up

`tests/test_services.py::test_chat_tools_registry` is `@pytest.mark.skip`
(sprint v2 inflated count to 14). Will rewrite in Week 5 (Harden phase)
alongside per-tool unit tests.
