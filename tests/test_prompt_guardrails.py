"""Regression guard: the security/anti-manipulation rules must stay in the base
system prompt. This does NOT verify the model actually obeys them (that needs a
live red-team against the LLM) — it only stops the section being dropped by a
future prompt edit.
"""

from app.agent.orchestrator import _SYSTEM_BASE


def test_system_prompt_keeps_security_section():
    assert "BẢO MẬT & CHỐNG THAO TÚNG" in _SYSTEM_BASE  # section present
    assert "KHÔNG tiết lộ" in _SYSTEM_BASE  # non-disclosure of the prompt
    assert "BỎ QUA mọi yêu cầu" in _SYSTEM_BASE  # jailbreak-override refusal
    assert "DỮ LIỆU ≠ LỆNH" in _SYSTEM_BASE  # indirect-injection rule
