"""
Per-request context object passed to every @function_tool via RunContextWrapper.

Tools use this to:
  - Read auth state (user_id, auth_token) without it leaking into the LLM prompt
  - Append raw listing payloads to `collected_listings` so the orchestrator can
    return them to the frontend after the agent run finishes
  - Append `action_links` — deterministic "open this page" chips (label + url).
    Chat can only ever show a handful of rows, so a tool that summarises a
    larger set points at the page that owns the full list. Built server-side
    from the tool's own arguments, never by the LLM, so the URL is always valid.

The dataclass is process-local; it never crosses the wire to the LLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    user_id: Optional[str] = None
    auth_token: Optional[str] = None
    collected_listings: List[Dict[str, Any]] = field(default_factory=list)
    action_links: List[Dict[str, str]] = field(default_factory=list)

    def add_action_link(self, label: str, url: str) -> None:
        """Register a deep link chip, de-duplicated by url (first label wins)."""
        if any(link["url"] == url for link in self.action_links):
            return
        self.action_links.append({"label": label, "url": url})
