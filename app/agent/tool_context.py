"""
Per-request context object passed to every @function_tool via RunContextWrapper.

Tools use this to:
  - Read auth state (user_id, auth_token) without it leaking into the LLM prompt
  - Append raw listing payloads to `collected_listings` so the orchestrator can
    return them to the frontend after the agent run finishes

The dataclass is process-local; it never crosses the wire to the LLM.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolContext:
    user_id: Optional[str] = None
    auth_token: Optional[str] = None
    collected_listings: List[Dict[str, Any]] = field(default_factory=list)
